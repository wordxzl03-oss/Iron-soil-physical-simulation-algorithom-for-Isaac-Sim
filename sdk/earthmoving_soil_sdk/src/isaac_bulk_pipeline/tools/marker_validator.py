"""Validate semantic USD marker positions and construct a Tool Frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


REQUIRED_MARKERS = (
    "ToolOrigin",
    "CuttingEdgeLeft",
    "CuttingEdgeCenter",
    "CuttingEdgeRight",
    "BottomRearLeft",
    "BottomRearRight",
    "SideTopLeft",
    "SideTopRight",
)


@dataclass(frozen=True)
class MarkerValidationResult:
    """Validated marker-derived ``T_reference_from_tool`` and diagnostics."""

    reference_from_tool: np.ndarray
    nominal_width_m: float
    diagnostics: dict[str, float]


def validate_marker_positions(
    marker_positions: Mapping[str, Sequence[float] | np.ndarray],
    *,
    origin_tolerance_m: float = 0.02,
    orthogonality_tolerance: float = 0.15,
) -> MarkerValidationResult:
    """Validate required marker XYZ points expressed in one metre reference frame.

    The returned matrix has Tool axes as columns and can be a world- or
    link-referenced transform depending on the caller's input positions.
    """

    missing = [name for name in REQUIRED_MARKERS if name not in marker_positions]
    if missing:
        raise ValueError(f"[MarkerValidator] missing required markers={missing}")
    points: dict[str, np.ndarray] = {}
    for name in REQUIRED_MARKERS:
        point = np.asarray(marker_positions[name], dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(
                f"[MarkerValidator] {name} must be finite XYZ metres; "
                f"shape={point.shape}, value={point!r}"
            )
        points[name] = point

    left = points["CuttingEdgeLeft"]
    center = points["CuttingEdgeCenter"]
    right = points["CuttingEdgeRight"]
    origin = points["ToolOrigin"]
    width_vector = right - left
    width = float(np.linalg.norm(width_vector))
    if width <= 1e-6:
        raise ValueError("[MarkerValidator] cutting-edge width is zero")
    endpoint_center = 0.5 * (left + right)
    origin_error = float(np.linalg.norm(origin - center))
    centerline_error = float(np.linalg.norm(center - endpoint_center))
    if origin_error > origin_tolerance_m:
        raise ValueError(
            "[MarkerValidator] ToolOrigin must coincide with CuttingEdgeCenter; "
            f"error_m={origin_error}, tolerance_m={origin_tolerance_m}"
        )
    if centerline_error > max(origin_tolerance_m, 0.02 * width):
        raise ValueError(
            "[MarkerValidator] CuttingEdgeCenter is not centred between endpoints; "
            f"error_m={centerline_error}, width_m={width}"
        )

    rear_center = 0.5 * (
        points["BottomRearLeft"] + points["BottomRearRight"]
    )
    x_axis = width_vector / width
    rear_to_mouth = center - rear_center
    raw_orthogonality = float(
        abs(np.dot(x_axis, rear_to_mouth)) / max(np.linalg.norm(rear_to_mouth), 1e-12)
    )
    if raw_orthogonality > orthogonality_tolerance:
        raise ValueError(
            "[MarkerValidator] rear-to-mouth direction is not sufficiently "
            f"orthogonal to cutting edge; normalized_dot={raw_orthogonality}"
        )
    y_vector = rear_to_mouth - np.dot(rear_to_mouth, x_axis) * x_axis
    depth = float(np.linalg.norm(y_vector))
    if depth <= 1e-6:
        raise ValueError("[MarkerValidator] rear and mouth markers are coincident")
    y_axis = y_vector / depth
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-12)
    top_center = 0.5 * (points["SideTopLeft"] + points["SideTopRight"])
    top_height = float(np.dot(top_center - origin, z_axis))
    if top_height <= 1e-4:
        raise ValueError(
            "[MarkerValidator] SideTop markers do not lie on Tool Frame +Z; "
            f"signed_height_m={top_height}. Check left/right marker labels."
        )

    transform = np.eye(4, dtype=np.float64)
    transform[:3, 0] = x_axis
    transform[:3, 1] = y_axis
    transform[:3, 2] = z_axis
    transform[:3, 3] = origin
    return MarkerValidationResult(
        reference_from_tool=transform,
        nominal_width_m=width,
        diagnostics={
            "origin_error_m": origin_error,
            "cutting_edge_center_error_m": centerline_error,
            "rear_to_mouth_depth_m": depth,
            "side_top_height_m": top_height,
            "edge_depth_normalized_dot": raw_orthogonality,
        },
    )
