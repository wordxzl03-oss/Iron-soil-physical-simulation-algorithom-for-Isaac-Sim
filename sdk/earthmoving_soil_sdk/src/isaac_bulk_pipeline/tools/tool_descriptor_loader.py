"""Load tool proxies from parameters, semantic USD markers or descriptor files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from ..config.loader import ToolConfig
from .marker_validator import REQUIRED_MARKERS, validate_marker_positions
from .bucket_geometry import BucketGeometryDescriptor, GeometryQuality, GeometrySource
from .tool_descriptor import ToolDescriptor


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack((values, np.ones(len(values), dtype=np.float64)))
    transformed = (transform @ homogeneous.T).T
    return transformed[:, :3] / transformed[:, 3:4]


class ToolDescriptorLoader:
    """Build a :class:`ToolDescriptor` without inspecting visual Mesh faces."""

    @staticmethod
    def load_config(path: str | Path) -> ToolConfig:
        """Load a standalone YAML file containing a top-level ``tool`` mapping."""

        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"[ToolDescriptorLoader] config not found: {source}")
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping) or not isinstance(loaded.get("tool"), Mapping):
            raise TypeError(
                f"[ToolDescriptorLoader] YAML must contain a tool mapping: {source}"
            )
        return ToolConfig.from_mapping(loaded["tool"], base_dir=source.parent)

    @classmethod
    def load(
        cls,
        config: ToolConfig,
        *,
        stage: Any | None = None,
        tool_link_prim: str | None = None,
    ) -> ToolDescriptor:
        """Load one descriptor using the configured source mode."""

        if config.descriptor_source == "parameters":
            return cls._from_parameters(config)
        if config.descriptor_source == "file":
            return cls._from_file(config)
        if stage is None or tool_link_prim is None:
            raise ValueError(
                "[ToolDescriptorLoader] markers source requires stage and "
                "tool_link_prim"
            )
        return cls._from_markers(config, stage, tool_link_prim)

    @staticmethod
    def save(descriptor: ToolDescriptor, path: str | Path) -> Path:
        """Persist an offline-extracted descriptor as JSON or NPZ."""

        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        mapping = descriptor.to_mapping()
        if destination.suffix.lower() == ".json":
            destination.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        elif destination.suffix.lower() == ".npz":
            np.savez_compressed(
                destination,
                cutting_edge_local=descriptor.cutting_edge_local,
                bottom_profile_local=descriptor.bottom_profile_local,
                left_boundary_local=descriptor.left_boundary_local,
                right_boundary_local=descriptor.right_boundary_local,
                interior_profile_local=np.empty((0, 3), dtype=np.float64)
                if descriptor.interior_profile_local is None
                else descriptor.interior_profile_local,
                tool_to_link_matrix=descriptor.tool_to_link_matrix,
                metadata_json=np.asarray(json.dumps(mapping)),
            )
        else:
            raise ValueError(
                "[ToolDescriptorLoader] descriptor file must end in .json or .npz; "
                f"path={destination}"
            )
        return destination

    @staticmethod
    def geometry_from_marker_positions(
        marker_positions: Mapping[str, np.ndarray],
        *,
        rated_capacity_m3: float | None = None,
    ) -> BucketGeometryDescriptor:
        """Build authoritative reduced-order geometry from the marker contract.

        Positions may be expressed in any one metre-based reference frame. The
        validator constructs the Tool Frame before the profile is assembled,
        making this path testable without USD/Isaac bindings.
        """

        geometry, _ = ToolDescriptorLoader._geometry_from_marker_positions(
            marker_positions,
            rated_capacity_m3=rated_capacity_m3,
        )
        return geometry

    @staticmethod
    def _geometry_from_marker_positions(
        marker_positions: Mapping[str, np.ndarray],
        *,
        rated_capacity_m3: float | None,
    ) -> tuple[BucketGeometryDescriptor, Any]:
        validation = validate_marker_positions(marker_positions)
        reference_from_tool = validation.reference_from_tool
        tool_from_reference = np.linalg.inv(reference_from_tool)
        local = {
            name: _transform_points(
                tool_from_reference,
                np.asarray(point, dtype=np.float64)[None, :],
            )[0]
            for name, point in marker_positions.items()
            if name in REQUIRED_MARKERS
        }
        rear_center = 0.5 * (local["BottomRearLeft"] + local["BottomRearRight"])
        top_center = 0.5 * (local["SideTopLeft"] + local["SideTopRight"])
        cutting = np.stack(
            (
                local["CuttingEdgeLeft"],
                local["CuttingEdgeCenter"],
                local["CuttingEdgeRight"],
            )
        )
        bottom = np.stack((rear_center, local["CuttingEdgeCenter"]))
        interior = np.stack((rear_center, top_center, local["CuttingEdgeCenter"]))
        top_edge = np.stack((local["SideTopLeft"], local["SideTopRight"]))
        geometry = BucketGeometryDescriptor.from_extruded_profile(
            cutting_edge_local=cutting,
            bottom_profile_local=bottom,
            interior_profile_local=interior,
            top_edge_local=top_edge,
            rated_capacity_m3=rated_capacity_m3,
            geometry_source=GeometrySource.MARKERS,
            geometry_quality=GeometryQuality.REDUCED_ORDER,
            metadata={"marker_contract": list(REQUIRED_MARKERS)},
        )
        return geometry, validation

    @staticmethod
    def _from_parameters(config: ToolConfig) -> ToolDescriptor:
        values = config.parameters
        try:
            width = float(values["nominal_width_m"])
            depth = float(values["mouth_depth_m"])
        except KeyError as exc:
            raise KeyError(
                f"[ToolDescriptorLoader:parameters] missing key={exc.args[0]!r}"
            ) from exc
        rear_height = float(values.get("rear_height_m", 1.0))
        interior_height = float(values.get("interior_height_m", 0.75 * rear_height))
        if min(width, depth, rear_height) <= 0.0:
            raise ValueError(
                "[ToolDescriptorLoader:parameters] width, mouth_depth and "
                f"rear_height must be positive; values={(width, depth, rear_height)}"
            )
        half_width = 0.5 * width
        cutting = np.asarray(
            [[-half_width, 0.0, 0.0], [0.0, 0.0, 0.0], [half_width, 0.0, 0.0]]
        )
        bottom = np.asarray([[0.0, -depth, 0.0], [0.0, 0.0, 0.0]])
        left = np.asarray(
            [
                [-half_width, -depth, 0.0],
                [-half_width, -depth, rear_height],
                [-half_width, -0.25 * depth, interior_height],
                [-half_width, 0.0, 0.0],
            ]
        )
        right = left.copy()
        right[:, 0] *= -1.0
        interior = np.asarray(
            [
                [0.0, -depth, 0.0],
                [0.0, -depth, rear_height],
                [0.0, -0.25 * depth, interior_height],
                [0.0, 0.0, 0.0],
            ]
        )
        transform = np.asarray(
            values.get("tool_to_link_matrix", np.eye(4)), dtype=np.float64
        )
        capacity = (
            config.nominal_capacity_m3
            if config.nominal_capacity_m3 is not None
            else values.get("nominal_capacity_m3")
        )
        top_edge = np.asarray(
            [[-half_width, -depth, rear_height], [half_width, -depth, rear_height]],
            dtype=np.float64,
        )
        geometry = BucketGeometryDescriptor.from_extruded_profile(
            cutting_edge_local=cutting,
            bottom_profile_local=bottom,
            interior_profile_local=interior,
            top_edge_local=top_edge,
            rated_capacity_m3=None if capacity is None else float(capacity),
            geometry_source=(
                GeometrySource.EXPLICIT_PROFILE
                if config.proxy_level == "L1"
                else GeometrySource.LEGACY_FALLBACK
            ),
            geometry_quality=(
                GeometryQuality.REDUCED_ORDER
                if config.proxy_level == "L1"
                else GeometryQuality.APPROXIMATE
            ),
            metadata={
                "profile_source": "parameterized explicit y-z profile",
                "fallback_reason": None
                if config.proxy_level == "L1"
                else "frozen FlatBottomQuadProxy_L0 compatibility contract",
            },
        )
        return ToolDescriptor(
            tool_type=str(values.get("tool_type", "bucket")),
            tool_frame_prim=config.tool_frame_prim,
            cutting_edge_local=cutting,
            bottom_profile_local=bottom,
            left_boundary_local=left,
            right_boundary_local=right,
            interior_profile_local=interior,
            nominal_width_m=width,
            nominal_capacity_m3=None if capacity is None else float(capacity),
            proxy_level=config.proxy_level,
            actual_proxy_type=config.actual_proxy_type,
            tool_to_link_matrix=transform,
            metadata={
                "descriptor_source": "parameters",
                "actual_proxy_type": config.actual_proxy_type,
                "proxy_geometry_used": "explicit closed extruded interior profile",
            },
            bucket_geometry=geometry,
        )

    @staticmethod
    def _from_file(config: ToolConfig) -> ToolDescriptor:
        assert config.descriptor_path is not None
        path = config.descriptor_path
        if not path.is_file():
            raise FileNotFoundError(
                f"[ToolDescriptorLoader:file] descriptor not found: {path}"
            )
        if path.suffix.lower() == ".json":
            mapping = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                if "metadata_json" not in archive.files:
                    raise ValueError(
                        f"[ToolDescriptorLoader:file] NPZ missing metadata_json: {path}"
                    )
                mapping = json.loads(str(archive["metadata_json"].item()))
                for name in (
                    "cutting_edge_local",
                    "bottom_profile_local",
                    "left_boundary_local",
                    "right_boundary_local",
                    "tool_to_link_matrix",
                ):
                    mapping[name] = np.array(archive[name], copy=True)
                interior = np.array(archive["interior_profile_local"], copy=True)
                mapping["interior_profile_local"] = None if not len(interior) else interior
        else:
            raise ValueError(
                f"[ToolDescriptorLoader:file] unsupported descriptor extension: {path}"
            )
        if mapping.get("schema_version") not in {
            "isaac-bulk-tool-descriptor-v1",
            "isaac-bulk-tool-descriptor-v2",
        }:
            raise ValueError(
                "[ToolDescriptorLoader:file] unsupported or missing schema_version; "
                f"path={path}, value={mapping.get('schema_version')!r}"
            )
        capacity = (
            config.nominal_capacity_m3
            if config.nominal_capacity_m3 is not None
            else mapping.get("nominal_capacity_m3")
        )
        metadata = dict(mapping.get("metadata", {}))
        metadata.update({"descriptor_source": "file", "descriptor_path": str(path)})
        geometry_mapping = mapping.get("bucket_geometry")
        geometry = (
            None
            if geometry_mapping is None
            else BucketGeometryDescriptor.from_mapping(geometry_mapping)
        )
        return ToolDescriptor(
            tool_type=str(mapping["tool_type"]),
            tool_frame_prim=config.tool_frame_prim,
            cutting_edge_local=np.asarray(mapping["cutting_edge_local"]),
            bottom_profile_local=np.asarray(mapping["bottom_profile_local"]),
            left_boundary_local=np.asarray(mapping["left_boundary_local"]),
            right_boundary_local=np.asarray(mapping["right_boundary_local"]),
            interior_profile_local=None
            if mapping.get("interior_profile_local") is None
            else np.asarray(mapping["interior_profile_local"]),
            nominal_width_m=float(mapping["nominal_width_m"]),
            nominal_capacity_m3=None if capacity is None else float(capacity),
            proxy_level=config.proxy_level or str(mapping.get("proxy_level", "L0")),
            actual_proxy_type=config.actual_proxy_type,
            tool_to_link_matrix=np.asarray(mapping["tool_to_link_matrix"]),
            metadata=metadata,
            bucket_geometry=geometry,
        )

    @staticmethod
    def _from_markers(
        config: ToolConfig,
        stage: Any,
        tool_link_prim: str,
    ) -> ToolDescriptor:
        assert config.marker_root is not None
        try:
            from pxr import UsdGeom
        except ImportError as exc:
            raise RuntimeError(
                "[ToolDescriptorLoader:markers] pxr bindings require Isaac Sim Python"
            ) from exc
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        if meters_per_unit <= 0.0 or not np.isfinite(meters_per_unit):
            raise ValueError(
                f"[ToolDescriptorLoader:markers] invalid metersPerUnit={meters_per_unit}"
            )
        cache = UsdGeom.XformCache()

        def world_matrix_m(prim_path: str) -> np.ndarray:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise ValueError(
                    "[ToolDescriptorLoader:markers] missing Prim; "
                    f"prim_path={prim_path}, marker_root={config.marker_root}"
                )
            # Gf matrix storage is row-vector based. The core pipeline uses
            # NumPy column vectors; transpose and convert translation to metres.
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64).T
            matrix[:3, 3] *= meters_per_unit
            return matrix

        marker_positions = {
            name: world_matrix_m(f"{config.marker_root}/{name}")[:3, 3]
            for name in REQUIRED_MARKERS
        }
        configured_frame_position = world_matrix_m(config.tool_frame_prim)[:3, 3]
        if not np.allclose(
            configured_frame_position,
            marker_positions["ToolOrigin"],
            atol=0.02,
            rtol=0.0,
        ):
            raise ValueError(
                "[ToolDescriptorLoader:markers] tool_frame_prim does not coincide "
                f"with ToolOrigin; prim_path={config.tool_frame_prim}"
            )
        geometry, validation = ToolDescriptorLoader._geometry_from_marker_positions(
            marker_positions,
            rated_capacity_m3=config.nominal_capacity_m3,
        )
        world_from_tool = validation.reference_from_tool
        world_from_link = world_matrix_m(tool_link_prim)
        link_from_tool = np.linalg.inv(world_from_link) @ world_from_tool
        cutting = geometry.cutting_edge_local
        bottom_profile = np.stack(
            (
                np.mean(geometry.bottom_plate_polygon_local[:2], axis=0),
                np.mean(geometry.lip_local, axis=0),
            )
        )
        interior = geometry.interior_profile_local
        left = geometry.left_side_wall_local
        right = geometry.right_side_wall_local
        return ToolDescriptor(
            tool_type=str(config.parameters.get("tool_type", "bucket")),
            tool_frame_prim=config.tool_frame_prim,
            cutting_edge_local=cutting,
            bottom_profile_local=bottom_profile,
            left_boundary_local=left,
            right_boundary_local=right,
            interior_profile_local=interior,
            nominal_width_m=validation.nominal_width_m,
            nominal_capacity_m3=config.nominal_capacity_m3,
            proxy_level=config.proxy_level,
            actual_proxy_type=config.actual_proxy_type,
            tool_to_link_matrix=link_from_tool,
            metadata={
                "descriptor_source": "markers",
                "actual_proxy_type": config.actual_proxy_type,
                "proxy_geometry_used": "validated marker-derived extruded profile",
                "marker_root": config.marker_root,
                "marker_diagnostics": validation.diagnostics,
            },
            bucket_geometry=geometry,
        )
