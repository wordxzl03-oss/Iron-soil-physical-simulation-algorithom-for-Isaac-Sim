"""Hard identity gate for a signed company-delivery GPU cycle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


FORMAL_TERRAIN_RELATIVE = Path(
    "continuous_heightmap_25m_closed_dataset/"
    "sequence_000_H0_track_pile_acceptance_m.csv"
)
FORMAL_MATERIAL_SCENARIO = "MOHAJERI_I3_AS_RECEIVED_LOW_STRESS_NEAR_MATCHED"


@dataclass(frozen=True)
class DeliveryPreflight:
    status: str
    runtime_backend: str
    state_authority: str
    grid_resolution_m: float
    grid_shape_yx: tuple[int, int]
    terrain_file: str
    terrain_file_exact_match: bool
    material_scenario: str
    material_scenario_exact_match: bool
    production_core: str
    full_field_transfer_disabled: bool
    root_pose_write_count: int
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["grid_shape_yx"] = list(self.grid_shape_yx)
        result["failures"] = list(self.failures)
        return result


def evaluate_delivery_preflight(
    *,
    repository_root: Path,
    runtime_backend: str,
    grid_resolution_m: float,
    grid_shape_yx: tuple[int, int],
    terrain_file: Path,
    material_scenario: str,
    production_core: str = "EarthmovingPhysicsCore",
    full_field_transfer_disabled: bool = True,
    root_pose_write_count: int = 0,
) -> DeliveryPreflight:
    """Reject identity drift before Kit/Isaac is constructed."""

    root = Path(repository_root).resolve()
    terrain = Path(terrain_file).resolve()
    formal = (root / FORMAL_TERRAIN_RELATIVE).resolve()
    checks = {
        "WRONG_BACKEND": str(runtime_backend).upper() == "GPU_RUNTIME",
        "WRONG_STATE_AUTHORITY": str(runtime_backend).upper() == "GPU_RUNTIME",
        "WRONG_GRID_RESOLUTION": float(grid_resolution_m) == 0.05,
        "WRONG_GRID_SHAPE": tuple(grid_shape_yx) == (701, 701),
        "WRONG_ACCEPTANCE_TERRAIN": terrain == formal and terrain.is_file(),
        "WRONG_MATERIAL_SCENARIO": material_scenario == FORMAL_MATERIAL_SCENARIO,
        "WRONG_PRODUCTION_CORE": production_core == "EarthmovingPhysicsCore",
        "FULL_FIELD_TRANSFER_ENABLED": bool(full_field_transfer_disabled),
        "ROOT_POSE_WRITE_PRESENT": int(root_pose_write_count) == 0,
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    return DeliveryPreflight(
        status="PASS" if not failures else "PRE_FLIGHT_REJECTED_NOT_ACCEPTED",
        runtime_backend=str(runtime_backend).upper(),
        state_authority=(
            "DEVICE" if str(runtime_backend).upper() == "GPU_RUNTIME" else "HOST"
        ),
        grid_resolution_m=float(grid_resolution_m),
        grid_shape_yx=tuple(int(x) for x in grid_shape_yx),
        terrain_file=str(terrain),
        terrain_file_exact_match=terrain == formal and terrain.is_file(),
        material_scenario=str(material_scenario),
        material_scenario_exact_match=(
            material_scenario == FORMAL_MATERIAL_SCENARIO
        ),
        production_core=str(production_core),
        full_field_transfer_disabled=bool(full_field_transfer_disabled),
        root_pose_write_count=int(root_pose_write_count),
        failures=failures,
    )
