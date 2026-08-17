import unittest

import numpy as np

from isaac_bulk_pipeline.performance.terrain_hierarchy_acceptance import (
    TerrainAcceptancePath,
    TerrainAcceptanceStatus,
    TerrainHierarchyAcceptanceReport,
    TerrainPathMetrics,
)


def metrics(path, status=TerrainAcceptanceStatus.PASS, **changes):
    values = dict(
        path=path,
        status=status,
        case_name="localized",
        resolution_m=0.05,
        simulated_time_s=1.0,
        wall_time_s=0.5,
        final_terrain_linf_error_m=1e-8,
        final_terrain_rmse_m=1e-9,
        volume_balance_error_m3=1e-11,
        momentum_balance_error_kg_m_s=np.zeros(3),
        soil_impulse_terrain_ns=np.ones(3),
        soil_impulse_error_ns=np.zeros(3),
        backend_identity="CPU_REFERENCE",
    )
    if path is TerrainAcceptancePath.GPU_OPTIMIZED:
        values.update(
            backend_identity="WARP_GPU_OPTIMIZED",
            arrays_resident_on_device=True,
            gpu_utilization_percent_mean=52.0,
        )
    values.update(changes)
    return TerrainPathMetrics(**values)


class TerrainHierarchyAcceptanceTests(unittest.TestCase):
    def test_report_requires_three_separate_paths(self):
        report = TerrainHierarchyAcceptanceReport.build(
            "localized",
            [
                metrics(TerrainAcceptancePath.CPU_REFERENCE),
                metrics(TerrainAcceptancePath.GPU_OPTIMIZED),
                metrics(TerrainAcceptancePath.LARGE_AVALANCHE_MOBILE_PATH),
            ],
        )
        self.assertEqual(report.overall_status, "PASS")
        self.assertEqual(set(report.to_dict()["results"]), set(item.value for item in TerrainAcceptancePath))
        self.assertEqual(report.to_dict()["results"]["GPU_OPTIMIZED"]["rtf"], 2.0)

    def test_cpu_fallback_cannot_claim_gpu_pass(self):
        with self.assertRaisesRegex(ValueError, "device-resident"):
            metrics(
                TerrainAcceptancePath.GPU_OPTIMIZED,
                arrays_resident_on_device=False,
                backend_identity="CPU_FALLBACK",
            )

    def test_gpu_unavailable_is_explicit_and_not_overall_pass(self):
        report = TerrainHierarchyAcceptanceReport.build(
            "localized",
            [
                metrics(TerrainAcceptancePath.CPU_REFERENCE),
                metrics(
                    TerrainAcceptancePath.GPU_OPTIMIZED,
                    TerrainAcceptanceStatus.UNAVAILABLE,
                    arrays_resident_on_device=False,
                    gpu_utilization_percent_mean=None,
                    backend_identity="WARP_UNAVAILABLE",
                    failure_reason="Warp CUDA device not available",
                ),
                metrics(TerrainAcceptancePath.LARGE_AVALANCHE_MOBILE_PATH),
            ],
        )
        self.assertEqual(report.overall_status, "NOT_PASSED")

    def test_formal_resolution_cannot_be_lowered(self):
        with self.assertRaisesRegex(ValueError, "lowered"):
            TerrainHierarchyAcceptanceReport.build(
                "localized",
                [
                    metrics(TerrainAcceptancePath.CPU_REFERENCE),
                    metrics(TerrainAcceptancePath.GPU_OPTIMIZED),
                    metrics(
                        TerrainAcceptancePath.LARGE_AVALANCHE_MOBILE_PATH,
                        resolution_m=0.10,
                    ),
                ],
            )


if __name__ == "__main__":
    unittest.main()
