"""Load Phase-1 pipeline settings from YAML without hidden unit conversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from ..terrain.terrain_grid import TerrainGrid


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"[Config] section {name!r} must be a YAML mapping")
    return value


@dataclass(frozen=True)
class ProjectConfig:
    """Run identity and output settings."""

    name: str
    seed: int
    output_dir: Path


@dataclass(frozen=True)
class TerrainConfig:
    """Authoritative height-map grid configuration, all distances in metres."""

    heightmap_path: Path
    nx: int
    ny: int
    dx_m: float
    dy_m: float
    origin_x_m: float
    origin_y_m: float
    terrain_prim_path: str
    update_rate_hz: float = 10.0
    source_axis_order: str = "yx"
    terrain_to_world_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64), compare=False
    )

    def to_grid(self, valid_mask: np.ndarray | None = None) -> TerrainGrid:
        """Construct the corresponding :class:`TerrainGrid`."""

        return TerrainGrid(
            nx=self.nx,
            ny=self.ny,
            dx=self.dx_m,
            dy=self.dy_m,
            origin_x=self.origin_x_m,
            origin_y=self.origin_y_m,
            terrain_prim_path=self.terrain_prim_path,
            terrain_to_world_matrix=self.terrain_to_world_matrix,
            valid_mask=valid_mask,
        )


@dataclass(frozen=True)
class MeshConfig:
    """Dynamic visual mesh update settings."""

    collision_enabled: bool = False
    update_normals: bool = True
    subdivision_scheme: str = "none"
    mesh_update_rate_hz: float = 10.0
    normal_update_rate_hz: float = 5.0
    double_sided: bool = True
    display_color_rgb: tuple[float, float, float] = (0.40, 0.25, 0.10)


@dataclass(frozen=True)
class SweepConfig:
    """Continuous tool-sweep sampling limits."""

    max_translation_step_grid_fraction: float = 0.5
    max_rotation_step_deg: float = 2.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SweepConfig":
        config = cls(
            max_translation_step_grid_fraction=float(
                values.get("max_translation_step_grid_fraction", 0.5)
            ),
            max_rotation_step_deg=float(values.get("max_rotation_step_deg", 2.0)),
        )
        if not 0.0 < config.max_translation_step_grid_fraction <= 1.0:
            raise ValueError(
                "[Config:Sweep] max_translation_step_grid_fraction must be in (0,1]"
            )
        if not 0.0 < config.max_rotation_step_deg <= 45.0:
            raise ValueError(
                "[Config:Sweep] max_rotation_step_deg must be in (0,45]"
            )
        return config


@dataclass(frozen=True)
class ExcavationConfig:
    """Height-field column-clipping settings, in metres."""

    mode: str = "column_clipping"
    minimum_cut_depth_m: float = 0.002
    minimum_terrain_height_m: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ExcavationConfig":
        config = cls(
            mode=str(values.get("mode", "column_clipping")),
            minimum_cut_depth_m=float(values.get("minimum_cut_depth_m", 0.002)),
            minimum_terrain_height_m=float(
                values.get("minimum_terrain_height_m", 0.0)
            ),
        )
        if config.mode != "column_clipping":
            raise ValueError(
                "[Config:Excavation] only mode='column_clipping' is supported"
            )
        if config.minimum_cut_depth_m < 0.0:
            raise ValueError(
                "[Config:Excavation] minimum_cut_depth_m must be non-negative"
            )
        if config.minimum_terrain_height_m < 0.0:
            raise ValueError(
                "[Config:Excavation] minimum_terrain_height_m must be non-negative"
            )
        return config


@dataclass(frozen=True)
class SolverConfig:
    """Minimum-slope relaxation settings and boundary semantics."""

    type: str = "minimum_slope"
    config_path: Path | None = None
    solve_trigger: str = "action_end"
    solve_every_n_steps: int = 1
    critical_angle_deg: float = 34.0
    max_iterations: int = 1000
    large_avalanche_iteration_threshold: int = 1000
    numerical_safety_max_iterations: int = 1_000_000
    tolerance: float = 1e-8
    sequence_enabled: bool = False
    sequence_stride: int = 250
    sequence_max_frames: int = 32
    sequence_dtype: str = "float32"
    sequence_memory_limit_mb: float = 256.0
    boundary_condition: str = "closed"
    boundary_height_m: float = 0.0
    conservation_tolerance_m3: float = 1e-8

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> "SolverConfig":
        merged = dict(values)
        config_value = values.get("config_path")
        config_path = None
        if config_value is not None:
            config_path = Path(str(config_value)).expanduser()
            if base_dir is not None and not config_path.is_absolute():
                config_path = (base_dir / config_path).resolve()
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"[Config:Solver] config_path not found: {config_path}"
                )
            external = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if not isinstance(external, Mapping):
                raise TypeError(
                    f"[Config:Solver] external YAML root must be a mapping: {config_path}"
                )
            external_values = external.get("solver", external)
            if not isinstance(external_values, Mapping):
                raise TypeError(
                    f"[Config:Solver] external solver section must be a mapping: {config_path}"
                )
            merged = {**external_values, **values}
        config = cls(
            type=str(merged.get("type", "minimum_slope")),
            config_path=config_path,
            solve_trigger=str(merged.get("solve_trigger", "action_end")),
            solve_every_n_steps=int(merged.get("solve_every_n_steps", 1)),
            critical_angle_deg=float(merged.get("critical_angle_deg", 34.0)),
            max_iterations=int(merged.get("max_iterations", 1000)),
            large_avalanche_iteration_threshold=int(
                merged.get("large_avalanche_iteration_threshold", 1000)
            ),
            numerical_safety_max_iterations=int(
                merged.get("numerical_safety_max_iterations", 1_000_000)
            ),
            tolerance=float(merged.get("tolerance", 1e-8)),
            sequence_enabled=bool(merged.get("sequence_enabled", False)),
            sequence_stride=int(
                merged.get(
                    "sequence_stride",
                    # Backward-compatible input only. New configs must use the
                    # explicit sequence_* names so the memory contract is clear.
                    merged.get("save_sequence_every", 250),
                )
            ),
            sequence_max_frames=int(merged.get("sequence_max_frames", 32)),
            sequence_dtype=str(merged.get("sequence_dtype", "float32")),
            sequence_memory_limit_mb=float(
                merged.get("sequence_memory_limit_mb", 256.0)
            ),
            boundary_condition=str(merged.get("boundary_condition", "closed")),
            boundary_height_m=float(merged.get("boundary_height_m", 0.0)),
            conservation_tolerance_m3=float(
                merged.get("conservation_tolerance_m3", 1e-8)
            ),
        )
        if config.type != "minimum_slope":
            raise ValueError("[Config:Solver] only type='minimum_slope' is available")
        if config.solve_trigger not in {"every_step", "every_n_steps", "action_end"}:
            raise ValueError(
                "[Config:Solver] solve_trigger must be every_step, every_n_steps or action_end"
            )
        if config.solve_every_n_steps < 1:
            raise ValueError("[Config:Solver] solve_every_n_steps must be >= 1")
        if not 0.0 < config.critical_angle_deg < 90.0:
            raise ValueError("[Config:Solver] critical_angle_deg must be in (0,90)")
        if config.max_iterations < 1 or config.sequence_stride < 1:
            raise ValueError(
                "[Config:Solver] max_iterations and sequence_stride must be >= 1"
            )
        if config.large_avalanche_iteration_threshold < 1:
            raise ValueError(
                "[Config:Solver] large_avalanche_iteration_threshold must be >= 1"
            )
        if (
            config.numerical_safety_max_iterations
            < config.large_avalanche_iteration_threshold
        ):
            raise ValueError(
                "[Config:Solver] numerical_safety_max_iterations must be >= "
                "large_avalanche_iteration_threshold"
            )
        if config.sequence_max_frames < 2:
            raise ValueError(
                "[Config:Solver] sequence_max_frames must be >= 2 so the "
                "pre/post states always fit"
            )
        if config.sequence_dtype not in {"float32", "float64"}:
            raise ValueError(
                "[Config:Solver] sequence_dtype must be float32 or float64"
            )
        if config.sequence_memory_limit_mb <= 0.0:
            raise ValueError(
                "[Config:Solver] sequence_memory_limit_mb must be positive"
            )
        if config.tolerance <= 0.0 or config.conservation_tolerance_m3 < 0.0:
            raise ValueError(
                "[Config:Solver] tolerances must be positive/non-negative"
            )
        if config.boundary_condition not in {"closed", "open"}:
            raise ValueError(
                "[Config:Solver] boundary_condition must be closed or open"
            )
        if config.boundary_height_m < 0.0:
            raise ValueError("[Config:Solver] boundary_height_m must be non-negative")
        return config


@dataclass(frozen=True)
class MaterialConfig:
    """Optional density metadata; geometric volume remains authoritative."""

    bulk_density_kg_m3: float | None = None
    density_is_estimated: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "MaterialConfig":
        density_value = values.get("bulk_density_kg_m3")
        config = cls(
            bulk_density_kg_m3=None
            if density_value is None
            else float(density_value),
            density_is_estimated=bool(values.get("density_is_estimated", True)),
        )
        if config.bulk_density_kg_m3 is not None and config.bulk_density_kg_m3 <= 0.0:
            raise ValueError(
                "[Config:Material] bulk_density_kg_m3 must be positive when provided"
            )
        return config


@dataclass(frozen=True)
class RobotConfig:
    """Robot and tool-link USD paths; no excavation semantics are included."""

    robot_root_prim: str
    articulation_root_prim: str
    tool_link_prim: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RobotConfig":
        """Build a robot configuration from one YAML mapping."""

        try:
            config = cls(
                robot_root_prim=str(values["robot_root_prim"]),
                articulation_root_prim=str(values["articulation_root_prim"]),
                tool_link_prim=str(values["tool_link_prim"]),
            )
        except KeyError as exc:
            raise KeyError(f"[Config:Robot] missing key={exc.args[0]!r}") from exc
        for name, path in (
            ("robot_root_prim", config.robot_root_prim),
            ("articulation_root_prim", config.articulation_root_prim),
            ("tool_link_prim", config.tool_link_prim),
        ):
            if not path.startswith("/"):
                raise ValueError(
                    f"[Config:Robot] {name} must be an absolute Prim path; value={path!r}"
                )
        return config


@dataclass(frozen=True)
class ToolConfig:
    """Tool descriptor source and source-specific settings."""

    descriptor_source: str
    tool_frame_prim: str
    marker_root: str | None = None
    descriptor_path: Path | None = None
    proxy_level: str = "L0"
    actual_proxy_type: str = "FlatBottomQuadProxy_L0"
    nominal_capacity_m3: float | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> "ToolConfig":
        """Build a tool configuration without inferring geometry from a Mesh."""

        try:
            source = str(values["descriptor_source"])
            frame = str(values["tool_frame_prim"])
        except KeyError as exc:
            raise KeyError(f"[Config:Tool] missing key={exc.args[0]!r}") from exc
        if source not in {"parameters", "markers", "file"}:
            raise ValueError(
                "[Config:Tool] descriptor_source must be parameters, markers or file; "
                f"value={source!r}"
            )
        if not frame.startswith("/"):
            raise ValueError(
                f"[Config:Tool] tool_frame_prim must be absolute; value={frame!r}"
            )
        marker_root_value = values.get("marker_root")
        marker_root = None if marker_root_value is None else str(marker_root_value)
        if source == "markers" and (marker_root is None or not marker_root.startswith("/")):
            raise ValueError(
                "[Config:Tool] markers source requires an absolute marker_root"
            )
        descriptor_value = values.get("descriptor_path")
        descriptor_path = None
        if descriptor_value is not None:
            descriptor_path = Path(descriptor_value).expanduser()
            if base_dir is not None and not descriptor_path.is_absolute():
                descriptor_path = (base_dir / descriptor_path).resolve()
        if source == "file" and descriptor_path is None:
            raise ValueError("[Config:Tool] file source requires descriptor_path")
        capacity_value = values.get("nominal_capacity_m3")
        capacity = None if capacity_value is None else float(capacity_value)
        if capacity is not None and capacity <= 0.0:
            raise ValueError("[Config:Tool] nominal_capacity_m3 must be positive")
        parameters = values.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise TypeError("[Config:Tool] parameters must be a mapping")
        proxy_level = str(values.get("proxy_level", "L0"))
        actual_proxy_type = str(
            values.get("actual_proxy_type", "FlatBottomQuadProxy_L0")
        )
        supported = {
            ("L0", "FlatBottomQuadProxy_L0"),
            ("L1", "ExtrudedProfileBucket_L1"),
        }
        if (proxy_level, actual_proxy_type) not in supported:
            raise ValueError(
                "[Config:Tool] unsupported tool proxy contract; expected legacy "
                "FlatBottomQuadProxy_L0 or ExtrudedProfileBucket_L1"
            )
        return cls(
            descriptor_source=source,
            tool_frame_prim=frame,
            marker_root=marker_root,
            descriptor_path=descriptor_path,
            proxy_level=proxy_level,
            actual_proxy_type=actual_proxy_type,
            nominal_capacity_m3=capacity,
            parameters=dict(parameters),
        )


@dataclass(frozen=True)
class PipelineConfig:
    """Typed Phase-1 settings plus untouched later-phase YAML sections."""

    project: ProjectConfig
    terrain: TerrainConfig
    mesh: MeshConfig
    robot: RobotConfig | None = None
    tool: ToolConfig | None = None
    sweep: SweepConfig = field(default_factory=SweepConfig)
    excavation: ExcavationConfig = field(default_factory=ExcavationConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    extra_sections: Mapping[str, Any] = field(default_factory=dict, compare=False)
    source_path: Path | None = None


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a pipeline YAML file.

    Relative paths are resolved against the YAML file's parent directory. No
    length, angle or height unit conversion is performed.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"[Config] YAML file not found: {source}")
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError(f"[Config] YAML root must be a mapping: {source}")

    project_raw = _section(loaded, "project")
    terrain_raw = _section(loaded, "terrain")
    mesh_raw = _section(loaded, "mesh")
    robot_raw = _section(loaded, "robot")
    tool_raw = _section(loaded, "tool")
    sweep_raw = _section(loaded, "sweep")
    excavation_raw = _section(loaded, "excavation")
    solver_raw = _section(loaded, "solver")
    material_raw = _section(loaded, "material")
    try:
        output_dir = Path(project_raw.get("output_dir", "outputs"))
        if not output_dir.is_absolute():
            output_dir = (source.parent / output_dir).resolve()
        project = ProjectConfig(
            name=str(project_raw["name"]),
            seed=int(project_raw.get("seed", 0)),
            output_dir=output_dir,
        )

        heightmap_path = Path(terrain_raw["heightmap_path"])
        if not heightmap_path.is_absolute():
            heightmap_path = (source.parent / heightmap_path).resolve()
        transform = np.asarray(
            terrain_raw.get("terrain_to_world_matrix", np.eye(4)),
            dtype=np.float64,
        )
        terrain = TerrainConfig(
            heightmap_path=heightmap_path,
            nx=int(terrain_raw["nx"]),
            ny=int(terrain_raw["ny"]),
            dx_m=float(terrain_raw["dx_m"]),
            dy_m=float(terrain_raw["dy_m"]),
            origin_x_m=float(terrain_raw["origin_x_m"]),
            origin_y_m=float(terrain_raw["origin_y_m"]),
            terrain_prim_path=str(terrain_raw["terrain_prim_path"]),
            update_rate_hz=float(terrain_raw.get("update_rate_hz", 10.0)),
            source_axis_order=str(terrain_raw.get("source_axis_order", "yx")),
            terrain_to_world_matrix=transform,
        )
        color = tuple(float(value) for value in mesh_raw.get("display_color_rgb", [0.40, 0.25, 0.10]))
        mesh = MeshConfig(
            collision_enabled=bool(mesh_raw.get("collision_enabled", False)),
            update_normals=bool(mesh_raw.get("update_normals", True)),
            subdivision_scheme=str(mesh_raw.get("subdivision_scheme", "none")),
            mesh_update_rate_hz=terrain.update_rate_hz,
            normal_update_rate_hz=float(mesh_raw.get("normal_update_rate_hz", 5.0)),
            double_sided=bool(mesh_raw.get("double_sided", True)),
            display_color_rgb=color,
        )
        robot = RobotConfig.from_mapping(robot_raw) if robot_raw else None
        tool = ToolConfig.from_mapping(tool_raw, base_dir=source.parent) if tool_raw else None
        sweep = SweepConfig.from_mapping(sweep_raw)
        excavation = ExcavationConfig.from_mapping(excavation_raw)
        solver = SolverConfig.from_mapping(solver_raw, base_dir=source.parent)
        material = MaterialConfig.from_mapping(material_raw)
    except KeyError as exc:
        raise KeyError(
            f"[Config] missing required key={exc.args[0]!r}; path={source}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"[Config] invalid value in {source}: {exc}") from exc

    if terrain.source_axis_order not in {"yx", "xy"}:
        raise ValueError(
            "[Config] terrain.source_axis_order must be 'yx' or explicit legacy "
            f"'xy'; value={terrain.source_axis_order!r}"
        )
    if terrain.update_rate_hz <= 0.0:
        raise ValueError("[Config] terrain.update_rate_hz must be positive")
    if mesh.normal_update_rate_hz <= 0.0:
        raise ValueError("[Config] mesh.normal_update_rate_hz must be positive")
    if mesh.subdivision_scheme != "none":
        raise ValueError(
            "[Config] mesh.subdivision_scheme must be 'none' for fixed topology"
        )
    if mesh.collision_enabled:
        raise ValueError(
            "[Config] dynamic terrain collision is disabled in Phase 1; "
            "mesh.collision_enabled must be false"
        )
    if len(mesh.display_color_rgb) != 3 or any(
        value < 0.0 or value > 1.0 for value in mesh.display_color_rgb
    ):
        raise ValueError("[Config] mesh.display_color_rgb must contain three values in [0,1]")

    # Construct once here so grid/transform errors report as configuration errors.
    terrain.to_grid()
    extras = {
        key: value
        for key, value in loaded.items()
        if key not in {
            "project",
            "terrain",
            "mesh",
            "robot",
            "tool",
            "sweep",
            "excavation",
            "solver",
            "material",
        }
    }
    return PipelineConfig(
        project=project,
        terrain=terrain,
        mesh=mesh,
        robot=robot,
        tool=tool,
        sweep=sweep,
        excavation=excavation,
        solver=solver,
        material=material,
        extra_sections=extras,
        source_path=source,
    )
