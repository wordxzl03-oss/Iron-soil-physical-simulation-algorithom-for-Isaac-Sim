import unittest

import numpy as np

from isaac_bulk_pipeline.soil_force.model import MobileMomentumBudget
from isaac_bulk_pipeline.soil_force.tool_mobile_contract import (
    ToolMobileFrameContractLedger,
    ToolMobileSubstepContract,
    external_acceleration_impulse_ns,
)


class ToolMobileMomentumContractTests(unittest.TestCase):
    def test_external_source_momentum_uses_depth_dual_area_density_and_dt(self):
        depth = np.asarray([[0.10, 0.20], [0.30, 0.40]])
        area = np.asarray([[0.25, 0.50], [0.75, 1.00]])
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
            depth,
            area,
            acceleration,
            bulk_density_kg_m3=density,
            dt_s=dt,
        )
        np.testing.assert_allclose(measured, expected, rtol=0.0, atol=1.0e-14)

    def test_synthetic_accepted_impulse_has_exact_opposite_machine_reaction(self):
        impulse = np.asarray([12.5, -3.25])
        record = ToolMobileSubstepContract(
            accepted_tool_to_mobile_impulse_xy_ns=impulse,
            measured_mobile_source_delta_xy_ns=impulse,
            contact_active=True,
            dt_sub_s=1.0 / 240.0,
        )
        np.testing.assert_array_equal(record.machine_reaction_impulse_xy_ns, -impulse)
        np.testing.assert_array_equal(record.source_residual_xy_ns, np.zeros(2))
        np.testing.assert_array_equal(record.action_reaction_residual_xy_ns, np.zeros(2))

    def test_multi_substep_accumulation_has_no_dt_double_scaling(self):
        impulses = (
            np.asarray([1.0, 2.0]),
            np.asarray([-0.25, 3.0]),
            np.asarray([4.5, -1.0]),
        )
        ledger = ToolMobileFrameContractLedger()
        for index, impulse in enumerate(impulses):
            ledger.append(
                ToolMobileSubstepContract(
                    accepted_tool_to_mobile_impulse_xy_ns=impulse,
                    measured_mobile_source_delta_xy_ns=impulse,
                    contact_active=True,
                    dt_sub_s=(index + 1) / 1000.0,
                )
            )
        expected = np.sum(impulses, axis=0)
        np.testing.assert_array_equal(
            ledger.accepted_tool_to_mobile_impulse_xy_ns, expected
        )
        np.testing.assert_array_equal(
            ledger.measured_mobile_source_delta_xy_ns, expected
        )
        np.testing.assert_array_equal(
            ledger.machine_reaction_impulse_xy_ns, -expected
        )
        np.testing.assert_array_equal(
            ledger.action_reaction_residual_xy_ns, np.zeros(2)
        )

    def test_zero_contact_cannot_leak_dynamic_impulse(self):
        record = ToolMobileSubstepContract(
            accepted_tool_to_mobile_impulse_xy_ns=np.zeros(2),
            measured_mobile_source_delta_xy_ns=np.zeros(2),
            contact_active=False,
            dt_sub_s=1.0 / 60.0,
        )
        ledger = ToolMobileFrameContractLedger([record])
        np.testing.assert_array_equal(
            ledger.accepted_tool_to_mobile_impulse_xy_ns, np.zeros(2)
        )
        np.testing.assert_array_equal(
            ledger.machine_reaction_impulse_xy_ns, np.zeros(2)
        )
        with self.assertRaisesRegex(ValueError, "without contact"):
            ToolMobileSubstepContract(
                accepted_tool_to_mobile_impulse_xy_ns=np.asarray([1.0, 0.0]),
                measured_mobile_source_delta_xy_ns=np.asarray([1.0, 0.0]),
                contact_active=False,
                dt_sub_s=1.0 / 60.0,
            )

    def test_machine_receives_exact_opposite_contact_moment_once_per_frame(self):
        budget = MobileMomentumBudget(
            momentum_before_terrain_kg_m_s=np.zeros(3),
            momentum_after_terrain_kg_m_s=np.asarray([4.0, -2.0, 0.5]),
            gravity_pressure_impulse_terrain_ns=np.zeros(3),
            basal_friction_impulse_terrain_ns=np.zeros(3),
            numerical_dissipative_impulse_terrain_ns=np.zeros(3),
            tool_impulse_on_mobile_terrain_ns=np.asarray([4.0, -2.0, 0.5]),
            tool_angular_impulse_on_mobile_about_tool_origin_terrain_nms=(
                np.asarray([0.75, -1.5, 6.0])
            ),
            integration_window_s=0.5,
        )
        np.testing.assert_array_equal(
            budget.soil_angular_impulse_on_tool_about_tool_origin_terrain_nms,
            np.asarray([-0.75, 1.5, -6.0]),
        )
        np.testing.assert_array_equal(
            budget.soil_torque_on_tool_about_tool_origin_terrain_nm,
            np.asarray([-1.5, 3.0, -12.0]),
        )


if __name__ == "__main__":
    unittest.main()
