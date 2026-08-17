"""Robot-independent computational tool proxy geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .bucket_geometry import (
    BucketGeometryDescriptor,
    GeometryQuality,
    GeometrySource,
)


def _points(values: np.ndarray, *, name: str, minimum_count: int = 2) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < minimum_count:
        raise ValueError(
            f"[ToolDescriptor] {name} must have shape (N,3), N>={minimum_count}; "
            f"received={array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"[ToolDescriptor] {name} contains NaN or Inf")
    result = np.ascontiguousarray(array.copy())
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ToolDescriptor:
    """Computational tool geometry expressed in the unified Tool Frame.

    Tool Frame contract:

    - origin: cutting-edge centre;
    - +X: left to right across the tool;
    - +Y: rear toward the mouth/cutting edge;
    - +Z: local upward/bottom normal.

    ``tool_to_link_matrix`` is ``T_link_from_tool`` in metres and maps these
    proxy points into the robot's configured tool-link local coordinates.
    """

    tool_type: str
    tool_frame_prim: str
    cutting_edge_local: np.ndarray
    bottom_profile_local: np.ndarray
    left_boundary_local: np.ndarray
    right_boundary_local: np.ndarray
    nominal_width_m: float
    interior_profile_local: np.ndarray | None = None
    nominal_capacity_m3: float | None = None
    proxy_level: str = "L0"
    actual_proxy_type: str = "FlatBottomQuadProxy_L0"
    tool_to_link_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
    bucket_geometry: BucketGeometryDescriptor | None = field(
        default=None,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.tool_type:
            raise ValueError("[ToolDescriptor] tool_type must not be empty")
        if not self.tool_frame_prim.startswith("/"):
            raise ValueError(
                "[ToolDescriptor] tool_frame_prim must be an absolute Prim path; "
                f"value={self.tool_frame_prim!r}"
            )
        if not np.isfinite(self.nominal_width_m) or self.nominal_width_m <= 0.0:
            raise ValueError(
                "[ToolDescriptor] nominal_width_m must be positive metres; "
                f"value={self.nominal_width_m}"
            )
        supported = {
            ("L0", "FlatBottomQuadProxy_L0"),
            ("L1", "ExtrudedProfileBucket_L1"),
        }
        if (self.proxy_level, self.actual_proxy_type) not in supported:
            raise ValueError(
                "[ToolDescriptor] unsupported proxy contract; "
                f"proxy_level={self.proxy_level!r}, "
                f"actual_proxy_type={self.actual_proxy_type!r}"
            )
        if self.nominal_capacity_m3 is not None and (
            not np.isfinite(self.nominal_capacity_m3)
            or self.nominal_capacity_m3 <= 0.0
        ):
            raise ValueError(
                "[ToolDescriptor] nominal_capacity_m3 must be positive when provided"
            )
        object.__setattr__(
            self,
            "cutting_edge_local",
            _points(self.cutting_edge_local, name="cutting_edge_local"),
        )
        object.__setattr__(
            self,
            "bottom_profile_local",
            _points(self.bottom_profile_local, name="bottom_profile_local"),
        )
        object.__setattr__(
            self,
            "left_boundary_local",
            _points(self.left_boundary_local, name="left_boundary_local"),
        )
        object.__setattr__(
            self,
            "right_boundary_local",
            _points(self.right_boundary_local, name="right_boundary_local"),
        )
        if self.interior_profile_local is not None:
            object.__setattr__(
                self,
                "interior_profile_local",
                _points(
                    self.interior_profile_local,
                    name="interior_profile_local",
                ),
            )

        transform = np.asarray(self.tool_to_link_matrix, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError(
                "[ToolDescriptor] tool_to_link_matrix must be finite shape (4,4); "
                f"received={transform.shape}"
            )
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12):
            raise ValueError(
                "[ToolDescriptor] tool_to_link_matrix must use affine column-vector form"
            )
        if abs(float(np.linalg.det(transform[:3, :3]))) < 1e-12:
            raise ValueError("[ToolDescriptor] tool_to_link_matrix is singular")
        transform = np.ascontiguousarray(transform.copy())
        transform.setflags(write=False)
        object.__setattr__(self, "tool_to_link_matrix", transform)
        object.__setattr__(self, "metadata", dict(self.metadata))

        geometry = self.bucket_geometry
        if geometry is None and self.interior_profile_local is not None and len(self.interior_profile_local) >= 3:
            top_center = self.interior_profile_local[
                int(np.argmax(self.interior_profile_local[:, 2]))
            ]
            top_edge = np.stack(
                (
                    np.asarray([self.cutting_edge_local[0, 0], top_center[1], top_center[2]]),
                    np.asarray([self.cutting_edge_local[-1, 0], top_center[1], top_center[2]]),
                )
            )
            geometry = BucketGeometryDescriptor.from_extruded_profile(
                cutting_edge_local=self.cutting_edge_local,
                bottom_profile_local=self.bottom_profile_local,
                interior_profile_local=self.interior_profile_local,
                top_edge_local=top_edge,
                rated_capacity_m3=self.nominal_capacity_m3,
                geometry_source=GeometrySource.LEGACY_FALLBACK,
                geometry_quality=GeometryQuality.APPROXIMATE,
                metadata={"fallback_reason": "legacy ToolDescriptor arrays without authoritative bucket_geometry"},
            )
        if geometry is not None:
            if not isinstance(geometry, BucketGeometryDescriptor):
                raise TypeError("[ToolDescriptor] bucket_geometry must be BucketGeometryDescriptor")
            if not np.allclose(
                geometry.cutting_edge_local[[0, -1]],
                self.cutting_edge_local[[0, -1]],
                atol=1.0e-9,
                rtol=0.0,
            ):
                raise ValueError("[ToolDescriptor] legacy cutting edge disagrees with bucket_geometry")
            if self.interior_profile_local is not None and len(self.interior_profile_local) >= 3 and not np.allclose(
                geometry.interior_profile_local,
                self.interior_profile_local,
                atol=1.0e-9,
                rtol=0.0,
            ):
                raise ValueError("[ToolDescriptor] interior profile disagrees with bucket_geometry")
            if self.nominal_capacity_m3 is not None and geometry.rated_capacity_m3 is not None and not np.isclose(
                self.nominal_capacity_m3,
                geometry.rated_capacity_m3,
                atol=1.0e-9,
                rtol=0.0,
            ):
                raise ValueError("[ToolDescriptor] rated capacities disagree")
        object.__setattr__(self, "bucket_geometry", geometry)

        measured_width = float(
            np.linalg.norm(self.cutting_edge_local[-1] - self.cutting_edge_local[0])
        )
        tolerance = max(1e-6, 0.02 * self.nominal_width_m)
        if abs(measured_width - self.nominal_width_m) > tolerance:
            raise ValueError(
                "[ToolDescriptor] cutting-edge width disagrees with nominal width; "
                f"measured={measured_width}, nominal={self.nominal_width_m}, "
                f"tolerance={tolerance}"
            )

    @property
    def proxy_radius_m(self) -> float:
        """Maximum proxy-point distance from Tool Frame origin, in metres."""

        arrays = [
            self.cutting_edge_local,
            self.bottom_profile_local,
            self.left_boundary_local,
            self.right_boundary_local,
        ]
        if self.interior_profile_local is not None:
            arrays.append(self.interior_profile_local)
        return float(max(np.linalg.norm(values, axis=1).max() for values in arrays))

    @property
    def geometric_capacity_m3(self) -> float | None:
        return None if self.bucket_geometry is None else self.bucket_geometry.geometric_capacity_m3

    @property
    def effective_capacity_m3(self) -> float | None:
        return None if self.bucket_geometry is None else self.bucket_geometry.effective_capacity_m3

    @property
    def geometry_source(self) -> str:
        return (
            GeometrySource.LEGACY_FALLBACK.value
            if self.bucket_geometry is None
            else self.bucket_geometry.geometry_source.value
        )

    @property
    def geometry_quality(self) -> str:
        return (
            GeometryQuality.APPROXIMATE.value
            if self.bucket_geometry is None
            else self.bucket_geometry.geometry_quality.value
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return a serialization-ready mapping with independent lists."""

        return {
            "schema_version": "isaac-bulk-tool-descriptor-v2",
            "tool_type": self.tool_type,
            "tool_frame_prim": self.tool_frame_prim,
            "cutting_edge_local": self.cutting_edge_local.tolist(),
            "bottom_profile_local": self.bottom_profile_local.tolist(),
            "left_boundary_local": self.left_boundary_local.tolist(),
            "right_boundary_local": self.right_boundary_local.tolist(),
            "interior_profile_local": None
            if self.interior_profile_local is None
            else self.interior_profile_local.tolist(),
            "nominal_width_m": self.nominal_width_m,
            "nominal_capacity_m3": self.nominal_capacity_m3,
            "proxy_level": self.proxy_level,
            "actual_proxy_type": self.actual_proxy_type,
            "tool_to_link_matrix": self.tool_to_link_matrix.tolist(),
            "metadata": dict(self.metadata),
            "bucket_geometry": None
            if self.bucket_geometry is None
            else self.bucket_geometry.to_mapping(),
        }
