"""Robot-independent computational tool geometry and kinematics."""

from .marker_validator import MarkerValidationResult, validate_marker_positions
from .bucket_geometry import (
    BucketGeometryDescriptor,
    CapacityMethod,
    GeometryQuality,
    GeometrySource,
)
from .tool_descriptor import ToolDescriptor
from .tool_descriptor_loader import ToolDescriptorLoader
from .tool_kinematics_adapter import ToolKinematicsAdapter, ToolState

__all__ = [
    "MarkerValidationResult",
    "BucketGeometryDescriptor",
    "CapacityMethod",
    "GeometryQuality",
    "GeometrySource",
    "ToolDescriptor",
    "ToolDescriptorLoader",
    "ToolKinematicsAdapter",
    "ToolState",
    "validate_marker_positions",
]
