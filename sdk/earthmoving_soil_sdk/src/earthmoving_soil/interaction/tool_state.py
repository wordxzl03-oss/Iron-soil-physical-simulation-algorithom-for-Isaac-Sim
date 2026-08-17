"""Public vehicle-independent tool schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from isaac_bulk_pipeline.config import ToolConfig
from isaac_bulk_pipeline.tools import ToolDescriptor, ToolDescriptorLoader


@dataclass(frozen=True)
class ToolGeometry:
    """Registered bucket geometry; ``descriptor`` is in its configured Tool Frame."""

    descriptor: ToolDescriptor

    @classmethod
    def from_descriptor(cls, descriptor: ToolDescriptor) -> "ToolGeometry":
        return cls(descriptor)

    @classmethod
    def parameterized_bucket(
        cls,
        *,
        width_m: float,
        mouth_depth_m: float,
        rear_height_m: float,
        capacity_m3: float,
        tool_frame_prim: str = "/World/Tool/SoilInteractionFrame",
        tool_to_link_matrix: np.ndarray | None = None,
    ) -> "ToolGeometry":
        """Build a non-proprietary L1 reduced-order bucket descriptor."""

        config = ToolConfig(
            descriptor_source="parameters",
            tool_frame_prim=tool_frame_prim,
            proxy_level="L1",
            actual_proxy_type="ExtrudedProfileBucket_L1",
            nominal_capacity_m3=float(capacity_m3),
            parameters={
                "tool_type": "bucket",
                "nominal_width_m": float(width_m),
                "mouth_depth_m": float(mouth_depth_m),
                "rear_height_m": float(rear_height_m),
                "nominal_capacity_m3": float(capacity_m3),
                "tool_to_link_matrix": (
                    np.eye(4) if tool_to_link_matrix is None else np.asarray(tool_to_link_matrix)
                ),
            },
        )
        return cls(ToolDescriptorLoader.load(config))


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite shape {shape}")
    result = np.ascontiguousarray(result.copy())
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ToolState:
    """Observed tool-link state in world coordinates and SI units.

    ``pose`` is ``T_world_from_tool_link`` using column vectors. Linear and
    angular velocities are world-frame m/s and rad/s. The SDK derives
    penetration, attack angle, failure depth and contact internally.
    """

    pose: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    timestamp: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose", _array(self.pose, (4, 4), "pose"))
        object.__setattr__(self, "linear_velocity", _array(self.linear_velocity, (3,), "linear_velocity"))
        object.__setattr__(self, "angular_velocity", _array(self.angular_velocity, (3,), "angular_velocity"))
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite seconds")
