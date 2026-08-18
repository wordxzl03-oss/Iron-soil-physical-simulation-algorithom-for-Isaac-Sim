"""Acceptance tests for the physical Resting-to-Mobile avalanche path."""

from __future__ import annotations

import numpy as np

from isaac_bulk_pipeline.bulk_interaction import (
    LITERATURE_REDUCED_ORDER_UNCALIBRATED,
    LargeAvalancheTransitionConfig,
    LargeAvalancheTransitionController,
    MobileLayerSolver,
)
from isaac_bulk_pipeline.bulk_state import MaterialScenario, TerrainVolumeIntegrator
from isaac_bulk_pipeline.terrain import TerrainGrid


def _material() -> MaterialScenario:
    return MaterialScenario(
        # The connected-region/ownership tests below exercise avalanche
        # mechanics, not cohesive onset.  Their historical 45 degree plane is
        # only physically unstable for this 0.05 m layer when cohesion is zero.
        name="cohesionless_avalanche_mechanics_fixture",
        assumed_bulk_density_kg_m3=2200.0,
        internal_friction_angle_deg=34.0,
        cohesion_proxy_pa=0.0,
        tool_friction_coefficient=0.45,
        start_angle_deg=38.0,
        stop_angle_deg=30.0,
        mobile_friction_coefficient=0.35,
    )


def _cohesive_material() -> MaterialScenario:
    return MaterialScenario(
        name="cohesive_semantic_equivalence_fixture",
        assumed_bulk_density_kg_m3=2200.0,
        internal_friction_angle_deg=34.0,
        cohesion_proxy_pa=1500.0,
        tool_friction_coefficient=0.45,
        start_angle_deg=38.0,
        stop_angle_deg=30.0,
        mobile_friction_coefficient=0.35,
    )


def _grid(shape: int = 21) -> tuple[TerrainGrid, TerrainVolumeIntegrator]:
    grid = TerrainGrid(shape, shape, 0.05, 0.05, 0.0, 0.0, "/World/Terrain")
    return grid, TerrainVolumeIntegrator.from_grid(grid)


def _steep_plane(grid: TerrainGrid, slope_deg: float = 45.0) -> np.ndarray:
    x = np.arange(grid.nx, dtype=np.float64) * grid.dx
    profile = 2.0 - np.tan(np.deg2rad(slope_deg)) * x
    return np.repeat(profile[None, :], grid.ny, axis=0)


def _fields(grid: TerrainGrid) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(grid.shape), np.zeros(grid.shape + (2,))


def _config(**changes: object) -> LargeAvalancheTransitionConfig:
    values: dict[str, object] = {
        "minimum_connected_cells": 16,
        "minimum_connected_area_m2": 0.02,
        "minimum_mobilizable_volume_m3": 0.001,
        "persistence_time_s": 0.10,
        "mobilization_depth_m": 0.05,
    }
    values.update(changes)
    return LargeAvalancheTransitionConfig.from_mapping(values)


def test_local_yield_is_immediate_while_persistence_only_classifies_large_event() -> None:
    grid, integrator = _grid()
    resting = _steep_plane(grid)
    mobile, momentum = _fields(grid)
    controller = LargeAvalancheTransitionController(_config())

    first = controller.observe_and_maybe_mobilize(
        resting, mobile, momentum, _material(), grid, integrator, 0.04
    )
    assert first.transitioned
    assert first.diagnostics.connected_extent_pass
    assert first.diagnostics.unstable_volume_pass
    assert first.diagnostics.mean_excess_start_deg > 0.0
    assert first.diagnostics.persistence_s == 0.04
    assert not first.diagnostics.persistence_pass
    assert (
        first.diagnostics.classification
        == controller.PERSISTING_LARGE_UNSTABLE_REGION
    )

    second = controller.observe_and_maybe_mobilize(
        first.H_resting_m,
        first.mobile_height_m,
        first.mobile_momentum_m2_s,
        _material(), grid, integrator, 0.04
    )
    assert not second.transitioned
    third = controller.observe_and_maybe_mobilize(
        second.H_resting_m,
        second.mobile_height_m,
        second.mobile_momentum_m2_s,
        _material(), grid, integrator, 0.02
    )
    # The currently owned tranche remains latched without real departure, but
    # persistence can still classify the sustained connected failure as large.
    assert not third.transitioned
    assert third.diagnostics.classification == controller.LARGE_AVALANCHE_MOBILE_PATH
    assert third.diagnostics.persistence_s == 0.10


def test_transfer_conserves_volume_and_starts_with_zero_horizontal_momentum() -> None:
    grid, integrator = _grid()
    resting = _steep_plane(grid)
    mobile, momentum = _fields(grid)
    controller = LargeAvalancheTransitionController(_config(persistence_time_s=0.01))
    before = integrator.integrate(resting + mobile)

    result = controller.observe_and_maybe_mobilize(
        resting, mobile, momentum, _material(), grid, integrator, 0.01
    )

    assert result.transitioned
    assert result.transferred_volume_m3 > 0.0
    assert abs(integrator.integrate(result.H_resting_m + result.mobile_height_m) - before) < 1e-12
    assert abs(
        integrator.integrate(resting) - integrator.integrate(result.H_resting_m)
        - result.transferred_volume_m3
    ) < 1e-12
    assert abs(
        integrator.integrate(result.mobile_height_m) - integrator.integrate(mobile)
        - result.transferred_volume_m3
    ) < 1e-12
    np.testing.assert_allclose(result.momentum_closure_error_kg_m_s, 0.0, atol=1e-11)
    np.testing.assert_allclose(
        result.momentum_after_kg_m_s - result.momentum_before_kg_m_s,
        result.gravity_initiation_impulse_kg_m_s,
        atol=1e-11,
    )
    # R->M is a mass-only state transition. Gravity/pressure acceleration is
    # applied subsequently by Mobile V2 over finite physical time.
    np.testing.assert_allclose(
        result.gravity_initiation_impulse_kg_m_s, np.zeros(2), atol=1e-12
    )


def test_latched_region_is_not_recounted_as_new_mobilizable_volume() -> None:
    grid, integrator = _grid()
    resting = _steep_plane(grid)
    mobile, momentum = _fields(grid)
    controller = LargeAvalancheTransitionController(
        _config(persistence_time_s=0.01)
    )
    first = controller.observe_and_maybe_mobilize(
        resting, mobile, momentum, _material(), grid, integrator, 0.01
    )
    assert first.transitioned
    assert first.diagnostics.classification == controller.LARGE_AVALANCHE_MOBILE_PATH

    after = controller.diagnose(
        first.H_resting_m,
        first.mobile_height_m,
        first.mobile_momentum_m2_s,
        _material(),
        grid,
        integrator,
    )
    assert after.unstable_cell_count > 0
    assert after.largest_connected_mobilizable_volume_m3 == 0.0
    # Event-scale volume is evaluated independent of ownership, so the event
    # classification does not disappear just because this tranche is latched.
    assert after.unstable_volume_pass
    assert not after.transition_ready


def test_latch_release_waits_for_complete_residual_static_event() -> None:
    grid, integrator = _grid()
    resting = _steep_plane(grid)
    mobile, momentum = _fields(grid)
    controller = LargeAvalancheTransitionController(
        _config(persistence_time_s=0.01)
    )
    first = controller.observe_and_maybe_mobilize(
        resting, mobile, momentum, _material(), grid, integrator, 0.01
    )
    assert first.transitioned
    assert np.any(controller._mobilized_latch_mask)

    flat = np.ones(grid.shape)
    controller.observe_and_maybe_mobilize(
        flat,
        np.zeros(grid.shape),
        np.zeros(grid.shape + (2,)),
        _material(),
        grid,
        integrator,
        0.01,
        release_settled_latches=False,
    )
    assert np.any(controller._mobilized_latch_mask)

    controller.observe_and_maybe_mobilize(
        flat,
        np.zeros(grid.shape),
        np.zeros(grid.shape + (2,)),
        _material(),
        grid,
        integrator,
        0.01,
        release_settled_latches=True,
    )
    assert not np.any(controller._mobilized_latch_mask)


def test_still_yielded_resting_retriggers_after_local_mobile_has_left() -> None:
    """A tranche latch cannot indefinitely hide current physical Y_start."""

    grid, integrator = _grid()
    resting = _steep_plane(grid)
    mobile, momentum = _fields(grid)
    material = MaterialScenario(
        name="cohesionless_retrigger_contract",
        assumed_bulk_density_kg_m3=2200.0,
        internal_friction_angle_deg=34.0,
        cohesion_proxy_pa=0.0,
        tool_friction_coefficient=0.45,
        start_angle_deg=38.0,
        stop_angle_deg=30.0,
        mobile_friction_coefficient=0.35,
    )
    controller = LargeAvalancheTransitionController(
        _config(persistence_time_s=0.01)
    )
    first = controller.observe_and_maybe_mobilize(
        resting, mobile, momentum, material, grid, integrator, 0.01
    )
    assert first.transitioned

    # Represent transport away from the activation cells without changing the
    # still-yielded Resting face.  The next tranche must become eligible; no
    # constitutive threshold or material parameter is changed.
    transport = MobileLayerSolver()
    transport.step(
        first.H_resting_m,
        first.mobile_height_m,
        first.mobile_momentum_m2_s,
        material,
        grid,
        integrator,
        0.01,
    )
    conservative_export = transport.last_conservative_export_m3
    activated = np.asarray(controller._mobilized_latch_mask, dtype=bool)
    assert np.any(
        conservative_export[activated]
        > controller.config.dry_tolerance_m
        * integrator.vertex_weights_m2[activated]
    )
    retrigger = controller.observe_and_maybe_mobilize(
        first.H_resting_m,
        np.zeros(grid.shape),
        np.zeros(grid.shape + (2,)),
        material,
        grid,
        integrator,
        0.01,
        release_settled_latches=True,
        conservative_export_cumulative_m3=conservative_export,
    )
    assert retrigger.diagnostics.unstable_cell_count > 0
    assert retrigger.diagnostics.largest_connected_mobilizable_volume_m3 > 0.0
    assert retrigger.transitioned


def test_local_redeposition_does_not_retrigger_the_same_owned_tranche() -> None:
    """Empty Mobile plus unchanged H_free is not evidence of transport away."""

    grid, integrator = _grid()
    resting = _steep_plane(grid)
    mobile, momentum = _fields(grid)
    material = MaterialScenario(
        name="cohesionless_local_redeposition_contract",
        assumed_bulk_density_kg_m3=2200.0,
        internal_friction_angle_deg=34.0,
        cohesion_proxy_pa=0.0,
        tool_friction_coefficient=0.45,
        start_angle_deg=38.0,
        stop_angle_deg=30.0,
        mobile_friction_coefficient=0.35,
    )
    controller = LargeAvalancheTransitionController(
        _config(persistence_time_s=0.01)
    )
    first = controller.observe_and_maybe_mobilize(
        resting, mobile, momentum, material, grid, integrator, 0.01
    )
    assert first.transitioned

    # Deposit every mobilized cell back where it came from.  Resting changes,
    # but authoritative H_free is exactly unchanged and there is no net
    # transport.  The original tranche must retain ownership.
    redeposited_resting = first.H_resting_m + first.mobile_height_m
    replay = controller.observe_and_maybe_mobilize(
        redeposited_resting,
        np.zeros(grid.shape),
        np.zeros(grid.shape + (2,)),
        material,
        grid,
        integrator,
        0.01,
        release_settled_latches=True,
        conservative_export_cumulative_m3=np.zeros(grid.shape),
    )
    assert replay.diagnostics.unstable_cell_count > 0
    assert replay.diagnostics.largest_connected_mobilizable_volume_m3 == 0.0
    assert not replay.transitioned
    assert np.any(controller._mobilized_latch_mask)


def test_gross_export_without_net_surface_departure_keeps_owned_tranche_latched() -> None:
    """Face traffic alone is not proof that the owned tranche left the cell."""

    grid, integrator = _grid()
    resting = _steep_plane(grid)
    mobile, momentum = _fields(grid)
    material = MaterialScenario(
        name="cohesionless_reversible_export_contract",
        assumed_bulk_density_kg_m3=2200.0,
        internal_friction_angle_deg=34.0,
        cohesion_proxy_pa=0.0,
        tool_friction_coefficient=0.45,
        start_angle_deg=38.0,
        stop_angle_deg=30.0,
        mobile_friction_coefficient=0.35,
    )
    controller = LargeAvalancheTransitionController(
        _config(persistence_time_s=0.01)
    )
    first = controller.observe_and_maybe_mobilize(
        resting, mobile, momentum, material, grid, integrator, 0.01
    )
    assert first.transitioned

    activated = np.asarray(controller._mobilized_latch_mask, dtype=bool)
    weights = integrator.vertex_weights_m2
    fake_gross_export = np.zeros(grid.shape, dtype=np.float64)
    fake_gross_export[activated] = (
        2.0 * controller.config.dry_tolerance_m * weights[activated]
    )

    # Put all Mobile straight back into Resting, so H_free is exactly the
    # activation-time owned surface.  Even with positive gross face-export
    # bookkeeping, there is no net tranche departure and ownership must stay.
    redeposited_resting = first.H_resting_m + first.mobile_height_m
    replay = controller.observe_and_maybe_mobilize(
        redeposited_resting,
        np.zeros(grid.shape),
        np.zeros(grid.shape + (2,)),
        material,
        grid,
        integrator,
        0.01,
        release_settled_latches=True,
        conservative_export_cumulative_m3=fake_gross_export,
    )
    assert replay.diagnostics.unstable_cell_count > 0
    assert replay.diagnostics.largest_connected_mobilizable_volume_m3 == 0.0
    assert not replay.transitioned
    assert np.any(controller._mobilized_latch_mask)


def test_disconnected_local_failures_use_local_yield_mobile_path() -> None:
    grid, integrator = _grid()
    resting = np.ones(grid.shape)
    # Four isolated spikes have steep local slopes but no qualifying connected
    # unstable region at the configured scale.
    resting[4, 4] += 0.2
    resting[4, 16] += 0.2
    resting[16, 4] += 0.2
    resting[16, 16] += 0.2
    mobile, momentum = _fields(grid)
    controller = LargeAvalancheTransitionController(
        _config(minimum_connected_cells=20, persistence_time_s=0.01)
    )

    result = controller.observe_and_maybe_mobilize(
        resting, mobile, momentum, _material(), grid, integrator, 1.0
    )
    assert result.transitioned
    assert result.diagnostics.unstable_cell_count > 0
    assert result.diagnostics.connected_region_count >= 4
    assert not result.diagnostics.connected_extent_pass
    assert (
        result.diagnostics.classification
        == controller.LOCAL_YIELD_MOBILE_PATH
    )
    assert result.transferred_volume_m3 > 0.0


def test_current_mobile_activity_and_settled_recovery_contract() -> None:
    grid, integrator = _grid()
    flat = np.ones(grid.shape)
    mobile, momentum = _fields(grid)
    controller = LargeAvalancheTransitionController(_config())
    quiet = controller.settled_diagnostic(
        flat, mobile, momentum, _material(), grid, integrator
    )
    assert quiet.settled
    assert quiet.reason == "TERRAIN_SETTLED"

    mobile[10, 10] = 0.03
    momentum[10, 10, 0] = mobile[10, 10] * 0.4
    active = controller.settled_diagnostic(
        flat, mobile, momentum, _material(), grid, integrator
    )
    assert active.current_mobile_active
    assert not active.settled
    assert not active.ready_for_final_minislope
    assert active.reason == "MOBILE_LAYER_ACTIVE"

    # A small constitutively yielded defect is still a physical Mobile event;
    # MiniSlope is reserved for residual geometry after Y_start is empty.
    residual = flat.copy()
    residual[10, 10] += 0.10
    mobile.fill(0.0)
    momentum.fill(0.0)
    recovery = controller.settled_diagnostic(
        residual, mobile, momentum, _material(), grid, integrator
    )
    assert not recovery.settled
    assert not recovery.ready_for_final_minislope
    assert not recovery.residual_static_relaxation_required
    assert recovery.reason == "LOCAL_YIELD_REQUIRES_MOBILE_PATH"


def test_uncalibrated_provenance_mapping_and_sensitivity_are_explicit() -> None:
    nominal = _config()
    cases = nominal.sensitivity_ensemble(0.20)
    assert [case.sensitivity_case for case in cases] == [
        "more_mobile_uncalibrated",
        "nominal_uncalibrated",
        "less_mobile_uncalibrated",
    ]
    assert all(case.parameter_basis == LITERATURE_REDUCED_ORDER_UNCALIBRATED for case in cases)
    assert cases[0].persistence_time_s < nominal.persistence_time_s < cases[2].persistence_time_s
    assert cases[0].mobilization_depth_m > nominal.mobilization_depth_m > cases[2].mobilization_depth_m

    try:
        LargeAvalancheTransitionConfig.from_mapping({"hidden_magic_threshold": 3})
    except ValueError as error:
        assert "unknown configuration fields" in str(error)
    else:
        raise AssertionError("unknown runtime/YAML keys must not be ignored")


def test_45_degree_cohesive_fixture_is_analytically_below_y_start() -> None:
    """The former angle-only fixture is cohesive-stable, not a physics failure."""

    grid, integrator = _grid()
    material = _cohesive_material()
    depth = 0.05
    slope = np.deg2rad(45.0)
    drive = material.assumed_bulk_density_kg_m3 * 9.81 * depth * np.sin(slope)
    normal = material.assumed_bulk_density_kg_m3 * 9.81 * depth * np.cos(slope)
    resistance = material.cohesion_proxy_pa + normal * np.tan(
        np.deg2rad(material.start_angle_deg)
    )
    assert abs(drive - 763.0389275784034) < 1.0e-6
    assert abs(resistance - 2096.1513465821067) < 1.0e-6
    assert drive - resistance < 0.0
    controller = LargeAvalancheTransitionController(
        _config(persistence_time_s=0.01, mobilization_depth_m=depth)
    )
    diagnostic = controller.diagnose(
        _steep_plane(grid), np.zeros(grid.shape), np.zeros(grid.shape + (2,)),
        material, grid, integrator,
    )
    assert diagnostic.unstable_cell_count == 0
    assert diagnostic.classification == controller.NO_LARGE_EVENT


def test_transition_exposes_measured_host_hotspot_breakdown() -> None:
    grid, integrator = _grid()
    controller = LargeAvalancheTransitionController(_config())
    mobile, momentum = _fields(grid)
    controller.observe_and_maybe_mobilize(
        _steep_plane(grid), mobile, momentum, _material(), grid, integrator, 0.02
    )
    profile = controller.last_profile_ms
    assert set(profile) == {
        "gradient",
        "connected_components",
        "diagnostics",
        "transfer",
        "total",
    }
    assert all(np.isfinite(value) and value >= 0.0 for value in profile.values())
    assert profile["total"] >= profile["gradient"]


def test_real_export_releases_owned_tranche_even_while_mobile_is_fast() -> None:
    """Departure evidence cannot be blocked by a low-speed latch condition."""

    grid, integrator = _grid()
    resting = _steep_plane(grid)
    mobile, momentum = _fields(grid)
    controller = LargeAvalancheTransitionController(
        _config(persistence_time_s=0.5)
    )
    first = controller.observe_and_maybe_mobilize(
        resting, mobile, momentum, _material(), grid, integrator, 0.01
    )
    assert first.transitioned
    activated = np.asarray(controller._mobilized_latch_mask, dtype=bool)
    assert np.any(activated)

    # Simulate conservative transport removing most of the owned Mobile while
    # the remainder is still moving well above the activity threshold.
    transported_mobile = np.array(first.mobile_height_m, copy=True)
    transported_mobile[activated] *= 0.25
    fast_momentum = np.zeros(grid.shape + (2,), dtype=np.float64)
    fast_momentum[..., 0] = transported_mobile * 1.0
    export = np.zeros(grid.shape, dtype=np.float64)
    export[activated] = (
        10.0 * controller.config.dry_tolerance_m
        * integrator.vertex_weights_m2[activated]
    )

    retrigger = controller.observe_and_maybe_mobilize(
        first.H_resting_m,
        transported_mobile,
        fast_momentum,
        _material(),
        grid,
        integrator,
        0.01,
        release_settled_latches=True,
        conservative_export_cumulative_m3=export,
    )
    assert retrigger.transferred_volume_m3 > 0.0


def test_removed_launch_velocity_parameters_are_rejected_as_stale_config() -> None:
    for stale in ("gravity_velocity_length_m", "velocity_efficiency"):
        try:
            LargeAvalancheTransitionConfig.from_mapping({stale: 0.5})
        except ValueError as exc:
            assert "unknown configuration fields" in str(exc)
        else:  # pragma: no cover - explicit contract
            raise AssertionError(f"stale field {stale} was silently accepted")
