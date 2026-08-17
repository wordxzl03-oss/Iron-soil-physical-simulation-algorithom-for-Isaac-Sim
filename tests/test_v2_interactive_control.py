import unittest
from pathlib import Path

from isaac_bulk_pipeline.runtime import (
    Interactive390FConfig,
    InteractiveControlModel,
    InteractiveRuntimeState,
    ResetLevel,
    RuntimeFailure,
    SoilForceMode,
)


ROOT = Path(__file__).resolve().parents[1]


class InteractiveControlTests(unittest.TestCase):
    def test_waits_for_user_and_runs_exact_requested_cycles(self):
        control = InteractiveControlModel()
        self.assertEqual(control.state, InteractiveRuntimeState.INITIALIZING)
        control.initialized()
        self.assertEqual(control.state, InteractiveRuntimeState.READY)
        self.assertFalse(control.physics_should_step)
        control.run_cycles(3)
        self.assertTrue(control.physics_should_step)
        control.cycle_completed()
        self.assertEqual(control.snapshot().current_cycle, 2)
        control.cycle_completed()
        self.assertEqual(control.snapshot().current_cycle, 3)
        control.cycle_completed()
        self.assertEqual(control.state, InteractiveRuntimeState.READY_NEXT_CYCLE)
        self.assertFalse(control.physics_should_step)

    def test_pause_resume_failure_emergency_and_reset_are_explicit(self):
        control = InteractiveControlModel()
        control.initialized()
        control.run_cycles(1)
        control.pause()
        self.assertEqual(control.state, InteractiveRuntimeState.PAUSED)
        control.resume()
        failure = RuntimeFailure("NAN", "non-finite force", 1, "penetrate", 0.5)
        control.fail(failure)
        self.assertEqual(control.state, InteractiveRuntimeState.FAILED)
        self.assertEqual(control.snapshot().failure.code, "NAN")
        control.reset_completed(ResetLevel.ALL)
        self.assertEqual(control.state, InteractiveRuntimeState.READY)
        control.emergency_stop()
        self.assertEqual(control.state, InteractiveRuntimeState.EMERGENCY_STOPPED)

    def test_physics_modes_change_only_while_ready(self):
        control = InteractiveControlModel()
        control.initialized()
        self.assertEqual(control.cycle_soil_force_mode(), SoilForceMode.NO_SOIL_FORCE)
        self.assertFalse(control.toggle_track_soil())
        control.run_cycles(1)
        with self.assertRaises(RuntimeError):
            control.cycle_soil_force_mode()
        with self.assertRaises(RuntimeError):
            control.toggle_track_soil()

    def test_formal_config_is_external_complete_and_defaults_to_gui_optimized(self):
        config = Interactive390FConfig.load(ROOT / "configs" / "390f_v2_interactive.yaml")
        self.assertFalse(config.headless)
        self.assertEqual(config.solver_backend, "OPTIMIZED")
        self.assertEqual(config.runtime_backend, "GPU_RUNTIME")
        self.assertEqual(config.soil_force_mode, SoilForceMode.FULL_SOIL_FORCE)
        self.assertTrue(config.track_soil_enabled)
        self.assertEqual(config.grid_shape, (701, 701))
        self.assertEqual(config.grid_spacing_m, 0.05)
        self.assertFalse(config.record_video)
        self.assertTrue(config.vehicle_asset.is_absolute())


if __name__ == "__main__":
    unittest.main()
