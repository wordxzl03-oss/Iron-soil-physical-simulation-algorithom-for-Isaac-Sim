"""Modular height-field bulk-material interaction pipeline for Isaac Sim.

The authoritative terrain representation throughout this package is a 2-D
NumPy array with the convention ``heightmap[row_y, column_x]`` in metres.
"""

from .terrain.heightmap_io import HeightmapIO, validate_heightmap
from .terrain.terrain_grid import TerrainGrid

__all__ = ["HeightmapIO", "TerrainGrid", "validate_heightmap"]
