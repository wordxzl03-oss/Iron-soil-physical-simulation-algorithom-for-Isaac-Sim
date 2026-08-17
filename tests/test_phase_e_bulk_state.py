import unittest
import hashlib
from pathlib import Path

import numpy as np
import yaml

from isaac_bulk_pipeline.bulk_state import (
    UNCALIBRATED_LABEL,
    BulkStateManager,
    ConservativeTransfer,
    MassLedger,
    MaterialParcel,
    MaterialScenario,
    PayloadState,
    Reservoir,
    SurfaceTopology,
    TerrainState,
    TerrainVolumeIntegrator,
    reservoir_volumes_from_state,
)
from tools.run_phase_e_acceptance import load_acceptance_config, run_acceptance


class PhaseEBulkStateTests(unittest.TestCase):
    @staticmethod
    def _material() -> MaterialScenario:
        return MaterialScenario(
            name="dry_loose_test",
            assumed_bulk_density_kg_m3=1000.0,
            internal_friction_angle_deg=34.0,
            cohesion_proxy_pa=0.0,
            tool_friction_coefficient=0.4,
            start_angle_deg=36.0,
            stop_angle_deg=31.0,
            mobile_friction_coefficient=0.5,
        )

    @classmethod
    def _state(
        cls,
        *,
        resting: float = 1.0,
        mobile: float = 0.0,
        payload_volume: float = 0.0,
        payload_capacity: float = 2.0,
        parcels=(),
        outflow: float = 0.0,
        timestamp: float = 0.0,
        action_index: int = 0,
    ) -> TerrainState:
        material = cls._material()
        return TerrainState(
            H_resting_m=np.full((3, 3), resting, dtype=np.float64),
            mobile_height_m=np.full((3, 3), mobile, dtype=np.float64),
            mobile_momentum_m2_s=np.zeros((3, 3, 2), dtype=np.float64),
            payload=PayloadState(
                volume_m3=payload_volume,
                capacity_m3=payload_capacity,
                assumed_bulk_density_kg_m3=(
                    material.assumed_bulk_density_kg_m3
                ),
                center_of_mass_bucket_frame_m=np.zeros(3),
            ),
            airborne_parcels=tuple(parcels),
            material=material,
            outflow_volume_m3=outflow,
            timestamp_s=timestamp,
            action_index=action_index,
        )

    @staticmethod
    def _integrator() -> TerrainVolumeIntegrator:
        return TerrainVolumeIntegrator(nx=3, ny=3, dx_m=1.0, dy_m=1.0)

    def test_payload_capacity_fill_ratio_and_estimated_mass(self) -> None:
        payload = self._state(payload_volume=0.5).payload
        self.assertEqual(payload.fill_ratio, 0.25)
        self.assertEqual(payload.remaining_capacity_m3, 1.5)
        self.assertEqual(payload.estimated_payload_mass_kg, 500.0)
        self.assertFalse(hasattr(payload, "true_mass"))
        with self.assertRaisesRegex(ValueError, "exceeds capacity"):
            payload.with_volume(2.1)

    def test_material_must_be_explicitly_uncalibrated(self) -> None:
        values = dict(
            name="bad",
            assumed_bulk_density_kg_m3=1000.0,
            internal_friction_angle_deg=34.0,
            cohesion_proxy_pa=0.0,
            tool_friction_coefficient=0.4,
            start_angle_deg=36.0,
            stop_angle_deg=31.0,
            mobile_friction_coefficient=0.5,
        )
        with self.assertRaisesRegex(ValueError, "explicitly uncalibrated"):
            MaterialScenario(**values, calibration_status="MEASURED")
        self.assertEqual(self._material().calibration_status, UNCALIBRATED_LABEL)

    def test_closed_resting_to_mobile_to_payload_balance(self) -> None:
        integrator = self._integrator()
        initial = self._state()
        manager = BulkStateManager(initial, integrator, boundary_condition="closed")

        mobile_state = self._state(
            resting=0.75,
            mobile=0.25,
            timestamp=1.0,
            action_index=1,
        )
        manager.commit_transfers(
            mobile_state,
            [
                ConservativeTransfer(
                    Reservoir.RESTING,
                    Reservoir.MOBILE,
                    1.0,
                    "phase_f_future_mobilization_fixture",
                )
            ],
        )
        payload_state = self._state(
            resting=0.75,
            mobile=0.125,
            payload_volume=0.5,
            timestamp=2.0,
            action_index=1,
        )
        manager.commit_transfers(
            payload_state,
            [
                ConservativeTransfer(
                    Reservoir.MOBILE,
                    Reservoir.PAYLOAD,
                    0.5,
                    "phase_f_future_intake_fixture",
                )
            ],
        )
        ledger = manager.ledger_snapshot()
        self.assertAlmostEqual(ledger.reservoirs.resting_m3, 3.0)
        self.assertAlmostEqual(ledger.reservoirs.mobile_m3, 0.5)
        self.assertAlmostEqual(ledger.reservoirs.payload_m3, 0.5)
        self.assertEqual(ledger.reservoirs.airborne_m3, 0.0)
        self.assertEqual(ledger.reservoirs.outflow_m3, 0.0)
        self.assertLess(ledger.balance.absolute_volume_error_m3, 1e-12)
        self.assertLess(ledger.balance.relative_volume_error, 1e-12)

    def test_airborne_reservoir_uses_coarse_parcel_volume(self) -> None:
        integrator = self._integrator()
        initial = self._state(payload_volume=0.5)
        manager = BulkStateManager(initial, integrator)
        parcel = MaterialParcel(
            parcel_id="parcel_0001",
            volume_m3=0.2,
            assumed_bulk_density_kg_m3=1000.0,
            position_world_m=np.asarray([0.0, 0.0, 2.0]),
            velocity_world_m_s=np.asarray([1.0, 0.0, 1.0]),
            timestamp_s=1.0,
            source="future_phase_g_spill_fixture",
        )
        next_state = self._state(
            payload_volume=0.3,
            parcels=(parcel,),
            timestamp=1.0,
        )
        manager.commit_transfers(
            next_state,
            [
                ConservativeTransfer(
                    Reservoir.PAYLOAD,
                    Reservoir.AIRBORNE,
                    0.2,
                    "future_phase_g_fixture",
                )
            ],
        )
        self.assertAlmostEqual(
            manager.ledger_snapshot().reservoirs.airborne_m3,
            0.2,
        )
        self.assertEqual(parcel.estimated_mass_kg, 200.0)

    def test_open_boundary_accounts_for_outflow(self) -> None:
        manager = BulkStateManager(
            self._state(),
            self._integrator(),
            boundary_condition="open",
        )
        next_state = self._state(resting=0.75, outflow=1.0, timestamp=1.0)
        manager.commit_transfers(
            next_state,
            [
                ConservativeTransfer(
                    Reservoir.RESTING,
                    Reservoir.OUTFLOW,
                    1.0,
                    "open_boundary_fixture",
                )
            ],
        )
        report = manager.ledger_snapshot().balance
        self.assertEqual(report.boundary_condition, "open")
        self.assertAlmostEqual(report.absolute_volume_error_m3, 0.0)
        self.assertAlmostEqual(report.estimated_absolute_mass_error_kg, 0.0)

    def test_closed_boundary_rejects_outflow_atomically(self) -> None:
        manager = BulkStateManager(self._state(), self._integrator())
        before = manager.snapshot()
        with self.assertRaisesRegex(ValueError, "closed boundary forbids outflow"):
            manager.commit_transfers(
                self._state(resting=0.75, outflow=1.0, timestamp=1.0),
                [
                    ConservativeTransfer(
                        Reservoir.RESTING,
                        Reservoir.OUTFLOW,
                        1.0,
                        "invalid_closed_fixture",
                    )
                ],
            )
        np.testing.assert_array_equal(manager.snapshot().H_resting_m, before.H_resting_m)
        self.assertEqual(manager.ledger_snapshot().transfer_count, 0)

    def test_capacity_and_negative_source_are_rejected(self) -> None:
        state = self._state(payload_capacity=0.5)
        ledger = MassLedger.initialize(
            state,
            self._integrator(),
            boundary_condition="closed",
        )
        with self.assertRaisesRegex(ValueError, "payload capacity"):
            ledger.transfer(
                ConservativeTransfer(
                    Reservoir.RESTING,
                    Reservoir.PAYLOAD,
                    0.6,
                    "capacity_test",
                )
            )
        with self.assertRaisesRegex(ValueError, "exceeds source"):
            ledger.transfer(
                ConservativeTransfer(
                    Reservoir.MOBILE,
                    Reservoir.PAYLOAD,
                    0.1,
                    "negative_source_test",
                )
            )
        reservoirs = ledger.snapshot().reservoirs
        self.assertEqual(reservoirs.mobile_m3, 0.0)
        self.assertEqual(reservoirs.payload_m3, 0.0)

    def test_unreported_spatial_change_is_rejected_atomically(self) -> None:
        manager = BulkStateManager(self._state(), self._integrator())
        with self.assertRaisesRegex(ValueError, "does not match explicit transfers"):
            manager.commit_transfers(
                self._state(resting=0.75, timestamp=1.0),
                [],
            )
        self.assertAlmostEqual(manager.snapshot().H_resting_m[0, 0], 1.0)
        self.assertEqual(manager.ledger_snapshot().transfer_count, 0)

    def test_state_rejects_negative_fields_and_momentum_on_dry_vertices(self) -> None:
        valid = self._state()
        with self.assertRaisesRegex(ValueError, "negative"):
            TerrainState(
                H_resting_m=-np.ones((3, 3)),
                mobile_height_m=valid.mobile_height_m,
                mobile_momentum_m2_s=valid.mobile_momentum_m2_s,
                payload=valid.payload,
                airborne_parcels=(),
                material=valid.material,
                outflow_volume_m3=0.0,
                timestamp_s=0.0,
                action_index=0,
            )
        momentum = np.zeros((3, 3, 2))
        momentum[1, 1, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "zero at dry"):
            TerrainState(
                H_resting_m=valid.H_resting_m,
                mobile_height_m=np.zeros((3, 3)),
                mobile_momentum_m2_s=momentum,
                payload=valid.payload,
                airborne_parcels=(),
                material=valid.material,
                outflow_volume_m3=0.0,
                timestamp_s=0.0,
                action_index=0,
            )

    def test_deep_snapshot_and_reset_restore_all_reservoirs(self) -> None:
        manager = BulkStateManager(self._state(), self._integrator())
        manager.commit_transfers(
            self._state(resting=0.75, mobile=0.25, timestamp=1.0),
            [
                ConservativeTransfer(
                    Reservoir.RESTING,
                    Reservoir.MOBILE,
                    1.0,
                    "reset_fixture",
                )
            ],
        )
        snapshot = manager.snapshot()
        snapshot.H_resting_m.setflags(write=True)
        snapshot.H_resting_m[0, 0] = 99.0
        self.assertEqual(manager.snapshot().H_resting_m[0, 0], 0.75)
        manager.reset()
        reset_state = manager.snapshot()
        reset_ledger = manager.ledger_snapshot()
        np.testing.assert_array_equal(reset_state.H_resting_m, np.ones((3, 3)))
        np.testing.assert_array_equal(reset_state.mobile_height_m, np.zeros((3, 3)))
        np.testing.assert_array_equal(
            reset_state.mobile_momentum_m2_s,
            np.zeros((3, 3, 2)),
        )
        self.assertEqual(reset_state.payload.volume_m3, 0.0)
        self.assertEqual(reset_state.airborne_parcels, ())
        self.assertEqual(reset_state.outflow_volume_m3, 0.0)
        self.assertEqual(reset_state.timestamp_s, 0.0)
        self.assertEqual(reset_state.action_index, 0)
        self.assertEqual(reset_ledger.transfer_count, 0)
        self.assertLess(reset_ledger.balance.absolute_volume_error_m3, 1e-12)

    def test_authoritative_ledger_rejects_bilinear_mode(self) -> None:
        bilinear = TerrainVolumeIntegrator(
            nx=3,
            ny=3,
            dx_m=1.0,
            dy_m=1.0,
            topology=SurfaceTopology.BILINEAR_TRAPEZOID,
        )
        with self.assertRaisesRegex(ValueError, "triangle_a_c"):
            reservoir_volumes_from_state(self._state(), bilinear)

    def test_phase_e_yaml_freezes_topology_and_future_transport_scope(self) -> None:
        path = Path(__file__).parents[1] / "configs" / "phase_e_bulk_state.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))[
            "phase_e_bulk_state"
        ]
        self.assertEqual(
            config["terrain_field"]["authoritative_surface_topology"],
            "triangle_a_c",
        )
        self.assertEqual(
            config["material_scenario"]["calibration_status"],
            UNCALIBRATED_LABEL,
        )
        self.assertEqual(
            config["phase_scope"]["mobile_transport_solver"],
            "future_phase_f",
        )
        self.assertTrue(config["phase_scope"]["explicit_conservative_transfer_api"])

    def test_acceptance_loads_consistent_phase_e_and_terrain_configs(self) -> None:
        root = Path(__file__).parents[1]
        phase_path = root / "configs" / "phase_e_bulk_state.yaml"
        terrain_path = root / "configs" / "project_25m.yaml"
        effective = load_acceptance_config(phase_path, terrain_path)
        self.assertEqual(effective.material.assumed_bulk_density_kg_m3, 2350.0)
        self.assertEqual(effective.payload_capacity_m3, 4.0)
        self.assertEqual(effective.boundary_condition, "closed")
        self.assertEqual(effective.absolute_tolerance_m3, 1e-10)
        self.assertEqual(effective.relative_tolerance, 1e-10)
        self.assertEqual((effective.nx, effective.ny), (701, 701))
        self.assertEqual((effective.dx_m, effective.dy_m), (0.05, 0.05))
        self.assertEqual((effective.span_x_m, effective.span_y_m), (35.0, 35.0))
        self.assertEqual(
            effective.phase_e_config_sha256,
            hashlib.sha256(phase_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            effective.terrain_config_sha256,
            hashlib.sha256(terrain_path.read_bytes()).hexdigest(),
        )

        heightmap = (
            root
            / "outputs"
            / "modular_25m_six_scoop"
            / "episode_0005"
            / "H_initial.npy"
        )
        result = run_acceptance(heightmap, effective)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            result["configuration"]["effective_values"][
                "assumed_bulk_density_kg_m3"
            ],
            2350.0,
        )
        self.assertEqual(
            result["configuration"]["effective_values"]["payload_capacity_m3"],
            4.0,
        )
        self.assertEqual(result["volume_profile"]["701"]["span_x_m"], 35.0)
        self.assertEqual(
            result["volume_profile"]["701"]["expected_volume_m3"],
            2450.0,
        )


if __name__ == "__main__":
    unittest.main()
