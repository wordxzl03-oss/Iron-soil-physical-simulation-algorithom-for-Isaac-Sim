"""External configuration schema for interactive/headless 390F V2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .interactive_control import SoilForceMode


@dataclass(frozen=True)
class Interactive390FConfig:
    config_path: Path
    vehicle_asset: Path
    articulation_root: str
    bucket_link: str
    lower_body: str
    left_track_body: str
    right_track_body: str
    world_anchor_joint: str
    mobile_base_enabled: bool
    auto_align_track_bottom_to_ground: bool
    track_ground_clearance_m: float
    initial_heightmap: Path
    bucket_descriptor: Path
    material_scenarios: Path
    material_scenario_id: str
    headless: bool
    solver_backend: str
    runtime_backend: str
    slope_backend: str
    active_tile_size: int
    large_avalanche_iteration_threshold: int
    numerical_safety_max_iterations: int
    minislope_round_budget_per_step: int
    minislope_tolerance_m: float
    large_avalanche_transition: Mapping[str, Any]
    soil_force_mode: SoilForceMode
    track_soil_enabled: bool
    track_footprint_length_m: float
    track_footprint_width_m: float
    nominal_track_belt_speed_m_s: float
    track_contact_gap_m: float
    track_soil: Mapping[str, Any]
    contact_chunk_size: int
    visual_chunk_size: int
    contact_update_hz: float
    cycle_count: int
    physics_dt_s: float
    grid_shape: tuple[int, int]
    grid_spacing_m: float
    dig_position_terrain_m: tuple[float, float, float]
    dump_position_terrain_m: tuple[float, float, float]
    terrain_translation_world_m: tuple[float, float, float]
    phase_timeout_s: float
    return_travel_timeout_s: float
    navigation_drive_heading_gate_rad: float
    phase_targets_rad: Mapping[str, np.ndarray]
    wait_for_user: bool
    debug_overlay_enabled: bool
    record_video: bool
    output_root: Path
    render_interval_sim_s: float
    viewport_width: int
    viewport_height: int
    controller: Mapping[str, Any]
    logging: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "Interactive390FConfig":
        config_path = Path(path).expanduser().resolve()
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("[V2Config] root must be a mapping")
        root = config_path.parents[1] if config_path.parent.name == "configs" else config_path.parent

        def resolve(value: str) -> Path:
            candidate = Path(value).expanduser()
            return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

        scene = data["scene"]
        terrain = data["terrain"]
        material = data["material"]
        physics = data["physics"]
        runtime = data["runtime"]
        operation = data["operation"]
        visualization = data["visualization"]
        output = data["output"]
        shape = tuple(int(value) for value in terrain["grid_shape_yx"])
        if len(shape) != 2 or shape != (701, 701):
            raise ValueError("[V2Config] formal V2 grid_shape_yx must be [701,701]")
        spacing = float(terrain["grid_spacing_m"])
        if not np.isclose(spacing, 0.05, atol=0.0, rtol=0.0):
            raise ValueError("[V2Config] formal V2 grid spacing must remain 0.05 m")
        backend = str(physics["solver_backend"]).upper()
        if backend not in {"REFERENCE", "OPTIMIZED"}:
            raise ValueError("[V2Config] solver_backend must be REFERENCE/OPTIMIZED")
        runtime_backend = str(
            physics.get("runtime_backend", "HOST_REFERENCE")
        ).upper()
        if runtime_backend not in {"HOST_REFERENCE", "GPU_RUNTIME"}:
            raise ValueError(
                "[V2Config] runtime_backend must be HOST_REFERENCE/GPU_RUNTIME"
            )
        slope_backend = str(
            physics.get(
                "slope_backend",
                "CPU_OPTIMIZED_COMPACT_TILE_FRONTIER",
            )
        ).upper()
        allowed_slope_backends = {
            "FULL_DOMAIN_REFERENCE",
            "CPU_OPTIMIZED_REACHABLE_BBOX_BASELINE",
            "CPU_OPTIMIZED_COMPACT_TILE_FRONTIER",
        }
        if slope_backend not in allowed_slope_backends:
            raise ValueError(
                f"[V2Config] unsupported slope_backend={slope_backend}"
            )
        active_tile_size = int(physics.get("active_tile_size", 64))
        large_threshold = int(
            physics.get("large_avalanche_iteration_threshold", 1_000)
        )
        safety_limit = int(
            physics.get("numerical_safety_max_iterations", 1_000_000)
        )
        minislope_round_budget = int(
            physics.get("minislope_round_budget_per_step", 1)
        )
        minislope_tolerance_m = float(
            physics.get("minislope_tolerance_m", 1.0e-8)
        )
        if active_tile_size < 4:
            raise ValueError("[V2Config] active_tile_size must be >= 4")
        if large_threshold < 1 or safety_limit < large_threshold:
            raise ValueError(
                "[V2Config] invalid avalanche threshold/numerical safety limit"
            )
        if minislope_round_budget < 1:
            raise ValueError(
                "[V2Config] minislope_round_budget_per_step must be >= 1"
            )
        if (
            not np.isfinite(minislope_tolerance_m)
            or minislope_tolerance_m <= 0.0
            or minislope_tolerance_m > 0.1 * spacing
        ):
            raise ValueError(
                "[V2Config] minislope_tolerance_m must be positive and no "
                "larger than 10% of formal grid spacing"
            )
        avalanche_transition = data.get("large_avalanche_transition", {})
        if not isinstance(avalanche_transition, Mapping):
            raise ValueError(
                "[V2Config] large_avalanche_transition must be a mapping"
            )
        cycle_count = int(operation["cycle_count"])
        if cycle_count < 1:
            raise ValueError("[V2Config] cycle_count must be >= 1")
        dt = float(physics["physics_dt_s"])
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("[V2Config] physics_dt_s must be positive")
        phase_timeout_s = float(operation.get("phase_timeout_s", 20.0))
        if not np.isfinite(phase_timeout_s) or phase_timeout_s <= 0.0:
            raise ValueError("[V2Config] phase_timeout_s must be finite/positive")
        return_travel_timeout_s = float(
            operation.get("return_travel_timeout_s", phase_timeout_s)
        )
        if (
            not np.isfinite(return_travel_timeout_s)
            or return_travel_timeout_s <= 0.0
        ):
            raise ValueError(
                "[V2Config] return_travel_timeout_s must be finite/positive"
            )
        navigation_drive_heading_gate_rad = float(
            np.deg2rad(operation.get("navigation_drive_heading_gate_deg", 35.0))
        )
        if not (
            np.isfinite(navigation_drive_heading_gate_rad)
            and 0.0 < navigation_drive_heading_gate_rad <= 0.5 * np.pi
        ):
            raise ValueError(
                "[V2Config] navigation_drive_heading_gate_deg must be in (0,90]"
            )
        phase_targets: dict[str, np.ndarray] = {}
        for name, values in operation["phase_targets_deg"].items():
            vector = np.asarray(values, dtype=np.float64)
            if vector.shape != (4,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"[V2Config] phase target {name} must be finite [4]")
            converted = np.ascontiguousarray(np.deg2rad(vector))
            converted.setflags(write=False)
            phase_targets[str(name)] = converted
        width = int(visualization.get("width", 1280))
        height = int(visualization.get("height", 720))
        if width < 320 or height < 240:
            raise ValueError("[V2Config] viewport dimensions are too small")
        track = data.get("track_soil", {})
        track_length = float(track.get("footprint_length_m", 6.17))
        track_width = float(track.get("footprint_width_m", 0.80))
        belt_speed = float(track.get("nominal_belt_speed_m_s", 1.20))
        contact_gap = float(track.get("contact_gap_m", 0.08))
        track_values = np.asarray([track_length, track_width, belt_speed, contact_gap])
        if not np.all(np.isfinite(track_values)) or np.any(track_values <= 0.0):
            raise ValueError("[V2Config] Track--Soil geometry/speed/gap must be positive")
        contact_chunk_size = int(terrain.get("contact_chunk_size", 64))
        visual_chunk_size = int(terrain.get("visual_chunk_size", 64))
        contact_update_hz = float(terrain.get("contact_update_hz", 2.0))
        if min(contact_chunk_size, visual_chunk_size) < 8 or not np.isfinite(contact_update_hz) or contact_update_hz <= 0.0:
            raise ValueError("[V2Config] invalid contact chunk size/update rate")
        return cls(
            config_path=config_path,
            vehicle_asset=resolve(scene["vehicle_asset"]),
            articulation_root=str(scene["articulation_root"]),
            bucket_link=str(scene["bucket_link"]),
            lower_body=str(scene["lower_body"]),
            left_track_body=str(scene["left_track_body"]),
            right_track_body=str(scene["right_track_body"]),
            world_anchor_joint=str(scene["world_anchor_joint"]),
            mobile_base_enabled=bool(scene.get("mobile_base_enabled", True)),
            auto_align_track_bottom_to_ground=bool(scene.get("auto_align_track_bottom_to_ground", True)),
            track_ground_clearance_m=float(scene.get("track_ground_clearance_m", 0.02)),
            initial_heightmap=resolve(terrain["initial_heightmap"]),
            bucket_descriptor=resolve(scene["bucket_descriptor"]),
            material_scenarios=resolve(material["scenario_file"]),
            material_scenario_id=str(material["scenario_id"]),
            headless=bool(runtime.get("headless", False)),
            solver_backend=backend,
            runtime_backend=runtime_backend,
            slope_backend=slope_backend,
            active_tile_size=active_tile_size,
            large_avalanche_iteration_threshold=large_threshold,
            numerical_safety_max_iterations=safety_limit,
            minislope_round_budget_per_step=minislope_round_budget,
            minislope_tolerance_m=minislope_tolerance_m,
            large_avalanche_transition=dict(avalanche_transition),
            soil_force_mode=SoilForceMode(str(physics["soil_force_mode"])),
            track_soil_enabled=bool(physics["track_soil_enabled"]),
            track_footprint_length_m=track_length,
            track_footprint_width_m=track_width,
            nominal_track_belt_speed_m_s=belt_speed,
            track_contact_gap_m=contact_gap,
            track_soil=dict(track),
            contact_chunk_size=contact_chunk_size,
            visual_chunk_size=visual_chunk_size,
            contact_update_hz=contact_update_hz,
            cycle_count=cycle_count,
            physics_dt_s=dt,
            grid_shape=shape,
            grid_spacing_m=spacing,
            dig_position_terrain_m=cls._point(operation["dig_position_terrain_m"], "dig"),
            dump_position_terrain_m=cls._point(operation["dump_position_terrain_m"], "dump"),
            terrain_translation_world_m=cls._point(terrain["terrain_translation_world_m"], "terrain_translation"),
            phase_timeout_s=phase_timeout_s,
            return_travel_timeout_s=return_travel_timeout_s,
            navigation_drive_heading_gate_rad=navigation_drive_heading_gate_rad,
            phase_targets_rad=phase_targets,
            wait_for_user=bool(runtime.get("wait_for_user", True)),
            debug_overlay_enabled=bool(visualization["debug_overlay_enabled"]),
            record_video=bool(visualization.get("record_video", False)),
            output_root=resolve(output["root"]),
            render_interval_sim_s=float(visualization["render_interval_sim_s"]),
            viewport_width=width,
            viewport_height=height,
            controller=dict(data.get("controller", {})),
            logging=dict(data.get("logging", {})),
        )

    @staticmethod
    def _point(value: Any, name: str) -> tuple[float, float, float]:
        point = np.asarray(value, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(f"[V2Config] {name} position must be finite [x,y,z]")
        return tuple(float(item) for item in point)
