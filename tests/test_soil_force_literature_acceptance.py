import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _acceptance_module():
    path = REPOSITORY_ROOT / "tools" / "run_soil_force_literature_acceptance.py"
    spec = importlib.util.spec_from_file_location("soil_force_literature_acceptance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SoilForceLiteratureAcceptanceTests(unittest.TestCase):
    def test_external_benchmark_runs_actual_failure_and_force_models(self) -> None:
        module = _acceptance_module()
        config = json.loads(
            (
                REPOSITORY_ROOT
                / "configs"
                / "literature"
                / "obermayr_2013_blade_benchmark.json"
            ).read_text(encoding="utf-8")
        )
        result = module._external_benchmark(config)
        self.assertEqual(result["reproducibility"], "FULLY_REPRODUCIBLE")
        self.assertEqual(result["parameter_assumptions"], [])
        self.assertFalse(result["force_cap_triggered"])
        self.assertEqual(
            result["comparison_status"],
            "COMPLETE_NO_PUBLISHED_OR_CALIBRATED_ERROR_THRESHOLD",
        )
        measured = np.concatenate(
            [case["experimental_horizontal_force_n"] for case in result["cases"]]
        )
        predicted = np.concatenate(
            [case["predicted_horizontal_force_n"] for case in result["cases"]]
        )
        error = predicted - measured
        self.assertEqual(len(measured), 8)
        self.assertTrue(np.all(np.isfinite(predicted)))
        self.assertTrue(np.all(predicted > 0.0))
        self.assertAlmostEqual(
            result["rmse_n"], float(np.sqrt(np.mean(error**2))), places=10
        )
        self.assertAlmostEqual(
            result["mean_absolute_relative_error"],
            float(np.mean(np.abs(error) / measured)),
            places=10,
        )

    def test_machine_readable_acceptance_separates_validation_claims(self) -> None:
        result = json.loads(
            (
                REPOSITORY_ROOT
                / "outputs"
                / "soil_force_literature_acceptance.json"
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(result["arbitrary_inertial_coefficient_used"])
        self.assertEqual(result["real_bucket_geometry_status"], "PASS")
        self.assertEqual(result["numerical_status"], "PASS")
        self.assertEqual(
            result["runtime_status"],
            "ISAAC_INTERFACE_READY_NOT_CLOSED_LOOP_RUNTIME_VALIDATED",
        )
        self.assertNotIn("FIELD", result["literature_validation_status"])
        self.assertEqual(
            result["uncertainty_force_envelope"]["case_count"], 81
        )
        self.assertEqual(
            result["uncertainty_force_envelope"]["force_cap_hit_rate"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
