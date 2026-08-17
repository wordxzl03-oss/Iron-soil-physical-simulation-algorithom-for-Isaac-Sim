from pathlib import Path

from isaac_bulk_pipeline.runtime.delivery_preflight import evaluate_delivery_preflight


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "continuous_heightmap_25m_closed_dataset/sequence_000_H0_track_pile_acceptance_m.csv"


def test_formal_gpu_cycle_identity_passes():
    result = evaluate_delivery_preflight(
        repository_root=ROOT,
        runtime_backend="GPU_RUNTIME",
        grid_resolution_m=0.05,
        grid_shape_yx=(701, 701),
        terrain_file=FORMAL,
        material_scenario="MOHAJERI_I3_AS_RECEIVED_LOW_STRESS_NEAR_MATCHED",
    )
    assert result.status == "PASS"
    assert result.state_authority == "DEVICE"
    assert result.failures == ()


def test_wrong_backend_and_terrain_are_rejected_before_isaac():
    result = evaluate_delivery_preflight(
        repository_root=ROOT,
        runtime_backend="HOST_REFERENCE",
        grid_resolution_m=0.05,
        grid_shape_yx=(701, 701),
        terrain_file=ROOT / "continuous_heightmap_25m_closed_dataset/sequence_000_H0_initial_m.csv",
        material_scenario="MOHAJERI_I3_AS_RECEIVED_LOW_STRESS_NEAR_MATCHED",
    )
    assert result.status == "PRE_FLIGHT_REJECTED_NOT_ACCEPTED"
    assert "WRONG_BACKEND" in result.failures
    assert "WRONG_ACCEPTANCE_TERRAIN" in result.failures
