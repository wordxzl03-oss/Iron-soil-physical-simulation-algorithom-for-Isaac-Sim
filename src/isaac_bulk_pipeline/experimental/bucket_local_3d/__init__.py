"""Experimental bucket-local 3D coarse-granular prototype.

Nothing in this package is production-authoritative yet. The ordinary-Python
helpers exist to make geometry/accounting testable independently of Isaac Sim.
"""

from .geometry import (
    BucketCavityGeometry,
    ParticleVolumeAccounting,
    inverse_transform_points,
    sample_rectangular_lattice,
    transform_points,
)

__all__ = [
    "BucketCavityGeometry",
    "ParticleVolumeAccounting",
    "inverse_transform_points",
    "sample_rectangular_lattice",
    "transform_points",
]
