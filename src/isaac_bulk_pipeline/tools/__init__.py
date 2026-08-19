"""Robot-independent computational tool geometry and kinematics."""

from .marker_validator import MarkerValidationResult, validate_marker_positions
from .bucket_geometry import (
    BucketGeometryDescriptor,
    CapacityMethod,
    GeometryQuality,
    GeometrySource,
)
from .bucket_physical_contact import (
    BucketPhysicalContactGeometry,
    assert_open_bucket_physical_contact,
    physical_bucket_contact_face_mask,
)
from .tool_descriptor import ToolDescriptor
from .tool_descriptor_loader import ToolDescriptorLoader
from .tool_kinematics_adapter import ToolKinematicsAdapter, ToolState

__all__ = [
    "MarkerValidationResult",
    "BucketGeometryDescriptor",
    "BucketPhysicalContactGeometry",
    "CapacityMethod",
    "GeometryQuality",
    "GeometrySource",
    "ToolDescriptor",
    "ToolDescriptorLoader",
    "ToolKinematicsAdapter",
    "ToolState",
    "assert_open_bucket_physical_contact",
    "physical_bucket_contact_face_mask",
    "validate_marker_positions",
]
