#!/usr/bin/env python3
"""Run reproducible CPython acceptance checks for the Phase-E state foundation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import yaml

from isaac_bulk_pipeline.bulk_state import (
    BULK_STATE_SCHEMA_VERSION,
    UNCALIBRATED_LABEL,
    BulkStateManager,
    ConservativeTransfer,
    MassLedger,
    MaterialScenario,
    PayloadState,
    Reservoir,
    TerrainState,
    TerrainVolumeIntegrator,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "phase_e_bulk_state.yaml"
DEFAULT_TERRAIN_CONFIG = REPOSITORY_ROOT / "configs" / "project_25m.yaml"
DEFAULT_HEIGHTMAP = (
    REPOSITORY_ROOT
    / "outputs"
    / "modular_25m_six_scoop"
    / "episode_0005"
    / "H_initial.npy"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs" / "phase_e_acceptance.json"


def _finite_positive(value: Any, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"[PhaseEAcceptance] {name} must be finite and > 0")
    return result


def _finite_nonnegative(value: Any, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(
            f"[PhaseEAcceptance] {name} must be finite and non-negative"
        )
    return result


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"[PhaseEAcceptance] {name} must be a YAML mapping")
    return value


def _load_yaml(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"[PhaseEAcceptance] config not found: {path}")
    return _mapping(
        yaml.safe_load(path.read_text(encoding="utf-8")),
        name=str(path),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path.resolve())


@dataclass(frozen=True)
class PhaseEAcceptanceConfig:
    phase_e_config_path: Path
    terrain_config_path: Path
    phase_e_config_sha256: str
    terrain_config_sha256: str
    material: MaterialScenario
    payload_capacity_m3: float
    boundary_condition: str
    absolute_tolerance_m3: float
    relative_tolerance: float
    nx: int
    ny: int
    dx_m: float
    dy_m: float
    terrain_solver_tolerance_m3: float

    @property
    def span_x_m(self) -> float:
        return float((self.nx - 1) * self.dx_m)

    @property
    def span_y_m(self) -> float:
        return float((self.ny - 1) * self.dy_m)


def load_acceptance_config(
    phase_e_config_path: Path,
    terrain_config_path: Path,
) -> PhaseEAcceptanceConfig:
    """Load both YAML files and reject inconsistent effective parameters."""

    phase_path = phase_e_config_path.resolve()
    terrain_path = terrain_config_path.resolve()
    phase_root = _mapping(
        _load_yaml(phase_path).get("phase_e_bulk_state"),
        name="phase_e_bulk_state",
    )
    terrain_root = _load_yaml(terrain_path)
    terrain = _mapping(terrain_root.get("terrain"), name="terrain")
    tool = _mapping(terrain_root.get("tool"), name="tool")
    project_material = _mapping(
        terrain_root.get("material"),
        name="project material",
    )
    solver = _mapping(terrain_root.get("solver"), name="solver")

    schema_version = str(phase_root.get("schema_version", ""))
    if schema_version != BULK_STATE_SCHEMA_VERSION:
        raise ValueError(
            "[PhaseEAcceptance] Phase-E schema mismatch: "
            f"expected={BULK_STATE_SCHEMA_VERSION}, received={schema_version}"
        )
    terrain_field = _mapping(
        phase_root.get("terrain_field"),
        name="phase_e terrain_field",
    )
    if terrain_field.get("semantic") != "vertex_field":
        raise ValueError("[PhaseEAcceptance] terrain semantic must be vertex_field")
    if terrain_field.get("axis_order") != "yx":
        raise ValueError("[PhaseEAcceptance] authoritative axis order must be yx")
    if terrain_field.get("authoritative_surface_topology") != "triangle_a_c":
        raise ValueError(
            "[PhaseEAcceptance] authoritative topology must be triangle_a_c"
        )

    material_values = _mapping(
        phase_root.get("material_scenario"),
        name="phase_e material_scenario",
    )
    material = MaterialScenario(
        name=str(material_values["name"]),
        assumed_bulk_density_kg_m3=material_values[
            "assumed_bulk_density_kg_m3"
        ],
        internal_friction_angle_deg=material_values[
            "internal_friction_angle_deg"
        ],
        cohesion_proxy_pa=material_values["cohesion_proxy_pa"],
        tool_friction_coefficient=material_values[
            "tool_friction_coefficient"
        ],
        start_angle_deg=material_values["start_angle_deg"],
        stop_angle_deg=material_values["stop_angle_deg"],
        mobile_friction_coefficient=material_values[
            "mobile_friction_coefficient"
        ],
        calibration_status=str(material_values["calibration_status"]),
    )
    if material.calibration_status != UNCALIBRATED_LABEL:
        raise ValueError("[PhaseEAcceptance] material must remain uncalibrated")

    payload = _mapping(phase_root.get("payload"), name="phase_e payload")
    payload_capacity_m3 = _finite_positive(
        payload.get("nominal_capacity_m3"),
        name="payload.nominal_capacity_m3",
    )
    project_capacity_m3 = _finite_positive(
        tool.get("nominal_capacity_m3"),
        name="project tool.nominal_capacity_m3",
    )
    if not np.isclose(
        payload_capacity_m3,
        project_capacity_m3,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "[PhaseEAcceptance] payload capacity differs between Phase-E and "
            f"terrain configs: {payload_capacity_m3} vs {project_capacity_m3}"
        )
    project_density = _finite_positive(
        project_material.get("bulk_density_kg_m3"),
        name="project material.bulk_density_kg_m3",
    )
    if not bool(project_material.get("density_is_estimated", False)):
        raise ValueError(
            "[PhaseEAcceptance] project density must be explicitly estimated"
        )
    if not np.isclose(
        material.assumed_bulk_density_kg_m3,
        project_density,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "[PhaseEAcceptance] assumed density differs between Phase-E and "
            f"terrain configs: {material.assumed_bulk_density_kg_m3} vs "
            f"{project_density}"
        )

    boundary_condition = str(phase_root.get("boundary_condition", ""))
    if boundary_condition not in {"closed", "open"}:
        raise ValueError("[PhaseEAcceptance] invalid Phase-E boundary condition")
    project_boundary = str(solver.get("boundary_condition", ""))
    if boundary_condition != project_boundary:
        raise ValueError(
            "[PhaseEAcceptance] boundary condition differs between configs: "
            f"{boundary_condition} vs {project_boundary}"
        )
    absolute_tolerance_m3 = _finite_nonnegative(
        phase_root.get("absolute_balance_tolerance_m3"),
        name="absolute_balance_tolerance_m3",
    )
    relative_tolerance = _finite_nonnegative(
        phase_root.get("relative_balance_tolerance"),
        name="relative_balance_tolerance",
    )
    terrain_solver_tolerance_m3 = _finite_nonnegative(
        solver.get("conservation_tolerance_m3"),
        name="solver.conservation_tolerance_m3",
    )
    if absolute_tolerance_m3 > terrain_solver_tolerance_m3:
        raise ValueError(
            "[PhaseEAcceptance] ledger absolute tolerance must be no looser than "
            "the terrain solver conservation tolerance"
        )

    nx = int(terrain.get("nx", 0))
    ny = int(terrain.get("ny", 0))
    dx_m = _finite_positive(terrain.get("dx_m"), name="terrain.dx_m")
    dy_m = _finite_positive(terrain.get("dy_m"), name="terrain.dy_m")
    if (nx, ny) != (701, 701):
        raise ValueError(
            f"[PhaseEAcceptance] expected configured 701x701; received={(ny, nx)}"
        )
    if not (
        np.isclose(dx_m, 0.05, rtol=0.0, atol=1e-12)
        and np.isclose(dy_m, 0.05, rtol=0.0, atol=1e-12)
    ):
        raise ValueError(
            f"[PhaseEAcceptance] expected configured 0.05 m spacing; dx={dx_m}, dy={dy_m}"
        )

    return PhaseEAcceptanceConfig(
        phase_e_config_path=phase_path,
        terrain_config_path=terrain_path,
        phase_e_config_sha256=_sha256(phase_path),
        terrain_config_sha256=_sha256(terrain_path),
        material=material,
        payload_capacity_m3=payload_capacity_m3,
        boundary_condition=boundary_condition,
        absolute_tolerance_m3=absolute_tolerance_m3,
        relative_tolerance=relative_tolerance,
        nx=nx,
        ny=ny,
        dx_m=dx_m,
        dy_m=dy_m,
        terrain_solver_tolerance_m3=terrain_solver_tolerance_m3,
    )


def _state(
    effective: PhaseEAcceptanceConfig,
    *,
    resting_height_m: float = 1.0,
    mobile_height_m: float = 0.0,
    payload_volume_m3: float = 0.0,
    outflow_volume_m3: float = 0.0,
    timestamp_s: float = 0.0,
) -> TerrainState:
    """Explicit 3x3/1m NUMERICAL fixture using YAML material/tool values."""

    return TerrainState(
        H_resting_m=np.full((3, 3), resting_height_m, dtype=np.float64),
        mobile_height_m=np.full((3, 3), mobile_height_m, dtype=np.float64),
        mobile_momentum_m2_s=np.zeros((3, 3, 2), dtype=np.float64),
        payload=PayloadState(
            volume_m3=payload_volume_m3,
            capacity_m3=effective.payload_capacity_m3,
            assumed_bulk_density_kg_m3=(
                effective.material.assumed_bulk_density_kg_m3
            ),
            center_of_mass_bucket_frame_m=np.zeros(3),
        ),
        airborne_parcels=(),
        material=effective.material,
        outflow_volume_m3=outflow_volume_m3,
        timestamp_s=timestamp_s,
        action_index=0,
    )


def _strict_triangle_reference(height: np.ndarray, dx_m: float, dy_m: float) -> float:
    """Independent expression for DynamicMeshAdapter's (a,b,c)/(a,c,d) mesh."""

    a = height[:-1, :-1].astype(np.float64, copy=False)
    b = height[:-1, 1:].astype(np.float64, copy=False)
    c = height[1:, 1:].astype(np.float64, copy=False)
    d = height[1:, :-1].astype(np.float64, copy=False)
    return float(
        np.sum(
            dx_m * dy_m / 6.0 * (2.0 * a + b + 2.0 * c + d),
            dtype=np.float64,
        )
    )


def _profile_resolutions(
    effective: PhaseEAcceptanceConfig,
) -> dict[str, dict[str, float | int]]:
    profile: dict[str, dict[str, float | int]] = {}
    span_x_m = effective.span_x_m
    span_y_m = effective.span_y_m
    run_count = 30
    for resolution in (128, 256, 512, 701):
        dx_m = span_x_m / (resolution - 1)
        dy_m = span_y_m / (resolution - 1)
        integrator = TerrainVolumeIntegrator(
            nx=resolution,
            ny=resolution,
            dx_m=dx_m,
            dy_m=dy_m,
        )
        height = np.full((resolution, resolution), 2.0, dtype=np.float32)
        timings_ms = []
        volumes = []
        for _ in range(run_count):
            start = perf_counter()
            volumes.append(integrator.integrate(height))
            timings_ms.append((perf_counter() - start) * 1_000.0)
        values = np.asarray(timings_ms, dtype=np.float64)
        expected_volume = 2.0 * span_x_m * span_y_m
        profile[str(resolution)] = {
            "run_count": run_count,
            "dx_m": dx_m,
            "dy_m": dy_m,
            "span_x_m": span_x_m,
            "span_y_m": span_y_m,
            "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.percentile(values, 50.0)),
            "p95_ms": float(np.percentile(values, 95.0)),
            "max_ms": float(np.max(values)),
            "volume_m3": float(volumes[-1]),
            "expected_volume_m3": expected_volume,
            "absolute_error_m3": float(abs(volumes[-1] - expected_volume)),
        }
    return profile


def run_acceptance(
    heightmap_path: Path,
    effective: PhaseEAcceptanceConfig,
) -> dict[str, object]:
    height = np.load(heightmap_path, allow_pickle=False)
    if height.shape != (effective.ny, effective.nx):
        raise ValueError(
            "Phase-E reference shape/config mismatch; "
            f"expected={(effective.ny, effective.nx)}, received={height.shape}"
        )
    if not np.all(np.isfinite(height)) or np.any(height < 0.0):
        raise ValueError("Phase-E reference H0 must be finite and non-negative")
    integrator_701 = TerrainVolumeIntegrator(
        nx=effective.nx,
        ny=effective.ny,
        dx_m=effective.dx_m,
        dy_m=effective.dy_m,
    )
    strict_volume = integrator_701.integrate(height)
    independent_volume = _strict_triangle_reference(
        height,
        effective.dx_m,
        effective.dy_m,
    )
    strict_error = abs(strict_volume - independent_volume)

    # Fixture geometry (3x3 vertices, 1m spacing) is NUMERICAL. Material,
    # capacity, boundary and tolerances come from the validated YAML pair.
    synthetic_integrator = TerrainVolumeIntegrator(
        nx=3,
        ny=3,
        dx_m=1.0,
        dy_m=1.0,
    )
    closed = BulkStateManager(
        _state(effective),
        synthetic_integrator,
        boundary_condition="closed",
        absolute_tolerance_m3=effective.absolute_tolerance_m3,
        relative_tolerance=effective.relative_tolerance,
    )
    closed.commit_transfers(
        _state(
            effective,
            resting_height_m=0.75,
            mobile_height_m=0.25,
            timestamp_s=1.0,
        ),
        [
            ConservativeTransfer(
                Reservoir.RESTING,
                Reservoir.MOBILE,
                1.0,
                "acceptance_closed_resting_to_mobile",
            )
        ],
    )
    closed_snapshot = closed.ledger_snapshot()
    closed.reset()
    closed_reset_state = closed.snapshot()
    closed_reset_ledger = closed.ledger_snapshot()
    reset_ok = bool(
        np.array_equal(closed_reset_state.H_resting_m, np.ones((3, 3)))
        and np.array_equal(closed_reset_state.mobile_height_m, np.zeros((3, 3)))
        and closed_reset_state.payload.volume_m3 == 0.0
        and closed_reset_state.airborne_parcels == ()
        and closed_reset_state.outflow_volume_m3 == 0.0
        and closed_reset_ledger.transfer_count == 0
    )

    opened = BulkStateManager(
        _state(effective),
        synthetic_integrator,
        boundary_condition="open",
        absolute_tolerance_m3=effective.absolute_tolerance_m3,
        relative_tolerance=effective.relative_tolerance,
    )
    opened.commit_transfers(
        _state(
            effective,
            resting_height_m=0.75,
            outflow_volume_m3=1.0,
            timestamp_s=1.0,
        ),
        [
            ConservativeTransfer(
                Reservoir.RESTING,
                Reservoir.OUTFLOW,
                1.0,
                "acceptance_open_outflow",
            )
        ],
    )
    open_snapshot = opened.ledger_snapshot()

    capacity_rejected = False
    no_negative_rejected = False
    fixture_area_m2 = synthetic_integrator.domain_area_m2
    capacity_source_height_m = (
        effective.payload_capacity_m3 + 1.0
    ) / fixture_area_m2
    ledger = MassLedger.initialize(
        _state(effective, resting_height_m=capacity_source_height_m),
        synthetic_integrator,
        boundary_condition="closed",
        absolute_tolerance_m3=effective.absolute_tolerance_m3,
        relative_tolerance=effective.relative_tolerance,
    )
    try:
        ledger.transfer(
            ConservativeTransfer(
                Reservoir.RESTING,
                Reservoir.PAYLOAD,
                effective.payload_capacity_m3 + 0.1,
                "acceptance_capacity_rejection",
            )
        )
    except ValueError as exc:
        capacity_rejected = "payload capacity" in str(exc)
    try:
        ledger.transfer(
            ConservativeTransfer(
                Reservoir.MOBILE,
                Reservoir.PAYLOAD,
                0.1,
                "acceptance_negative_source_rejection",
            )
        )
    except ValueError as exc:
        no_negative_rejected = "exceeds source" in str(exc)

    profile = _profile_resolutions(effective)
    profile_volumes_ok = all(
        item["absolute_error_m3"] <= 1e-9 for item in profile.values()
    )
    balance_tolerance = effective.absolute_tolerance_m3
    checks = {
        "config_schema": True,
        "config_topology_triangle_a_c": True,
        "config_material_and_capacity_consistent": True,
        "config_boundary_and_tolerances_consistent": True,
        "configured_grid_701_at_0p05_m": True,
        "real_h0_shape_matches_config": height.shape
        == (effective.ny, effective.nx),
        "real_h0_finite_nonnegative": bool(
            np.all(np.isfinite(height)) and np.all(height >= 0.0)
        ),
        "strict_triangle_matches_independent_formula": strict_error <= 1e-10,
        "closed_balance": (
            closed_snapshot.balance.absolute_volume_error_m3 <= balance_tolerance
        ),
        "open_balance_includes_outflow": (
            open_snapshot.balance.absolute_volume_error_m3 <= balance_tolerance
            and open_snapshot.reservoirs.outflow_m3 == 1.0
        ),
        "payload_capacity_rejected": capacity_rejected,
        "negative_reservoir_rejected": no_negative_rejected,
        "deep_reset": reset_ok,
        "required_resolution_volumes": profile_volumes_ok,
    }
    return {
        "schema_version": "phase_e.acceptance.v2",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "configuration": {
            "phase_e": {
                "path": _relative_path(effective.phase_e_config_path),
                "sha256": effective.phase_e_config_sha256,
            },
            "terrain": {
                "path": _relative_path(effective.terrain_config_path),
                "sha256": effective.terrain_config_sha256,
            },
            "effective_values": {
                "bulk_state_schema_version": BULK_STATE_SCHEMA_VERSION,
                "surface_topology": "triangle_a_c",
                "material_scenario_name": effective.material.name,
                "calibration_status": effective.material.calibration_status,
                "assumed_bulk_density_kg_m3": (
                    effective.material.assumed_bulk_density_kg_m3
                ),
                "payload_capacity_m3": effective.payload_capacity_m3,
                "boundary_condition": effective.boundary_condition,
                "absolute_balance_tolerance_m3": (
                    effective.absolute_tolerance_m3
                ),
                "relative_balance_tolerance": effective.relative_tolerance,
                "terrain_solver_conservation_tolerance_m3": (
                    effective.terrain_solver_tolerance_m3
                ),
                "nx": effective.nx,
                "ny": effective.ny,
                "dx_m": effective.dx_m,
                "dy_m": effective.dy_m,
                "span_x_m": effective.span_x_m,
                "span_y_m": effective.span_y_m,
            },
            "synthetic_transfer_fixture": {
                "shape": [3, 3],
                "dx_m": 1.0,
                "dy_m": 1.0,
                "geometry_provenance": "NUMERICAL",
                "material_and_capacity_source": "phase_e YAML",
            },
        },
        "scope": {
            "implemented": (
                "state, topology-consistent volume, explicit transfers, ledger"
            ),
            "not_implemented_until_phase_f": [
                "failure_zone",
                "mobile_transport",
                "bucket_intake_flux",
                "deposition",
            ],
        },
        "reference": {
            "path": _relative_path(heightmap_path),
            "shape": list(height.shape),
            "dtype": str(height.dtype),
            "dx_m": effective.dx_m,
            "dy_m": effective.dy_m,
            "topology": "triangle_a_c",
            "strict_volume_m3": strict_volume,
            "independent_formula_volume_m3": independent_volume,
            "absolute_error_m3": strict_error,
        },
        "closed_balance": {
            "reservoirs": {
                "resting_m3": closed_snapshot.reservoirs.resting_m3,
                "mobile_m3": closed_snapshot.reservoirs.mobile_m3,
                "payload_m3": closed_snapshot.reservoirs.payload_m3,
                "airborne_m3": closed_snapshot.reservoirs.airborne_m3,
                "outflow_m3": closed_snapshot.reservoirs.outflow_m3,
            },
            "absolute_error_m3": closed_snapshot.balance.absolute_volume_error_m3,
            "relative_error": closed_snapshot.balance.relative_volume_error,
        },
        "open_balance": {
            "outflow_m3": open_snapshot.reservoirs.outflow_m3,
            "absolute_error_m3": open_snapshot.balance.absolute_volume_error_m3,
            "relative_error": open_snapshot.balance.relative_volume_error,
        },
        "checks": checks,
        "volume_profile": profile,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--terrain-config",
        type=Path,
        default=DEFAULT_TERRAIN_CONFIG,
    )
    parser.add_argument("--heightmap", type=Path, default=DEFAULT_HEIGHTMAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    effective = load_acceptance_config(args.config, args.terrain_config)
    result = run_acceptance(args.heightmap.resolve(), effective)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"PHASE_E_ACCEPTANCE_{result['status']} "
        f"output={output} strict_volume_m3="
        f"{result['reference']['strict_volume_m3']:.12f}"
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
