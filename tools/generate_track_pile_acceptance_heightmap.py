#!/usr/bin/env python3
"""Generate a 0.05 m track-on-loose-ore approach-toe acceptance terrain.

The production pile is preserved.  A shallow, closed, smooth ore toe extends
toward the staged tracks so the real chassis can exercise TrackSoil and feed
the resulting rut back to contact without moving the machine by teleportation.
This is an explicit acceptance scenario, not a site-calibrated haul-road model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "continuous_heightmap_25m_closed_dataset/sequence_000_H0_initial_m.csv"
OUTPUT = ROOT / "continuous_heightmap_25m_closed_dataset/sequence_000_H0_track_pile_acceptance_m.csv"


def main() -> None:
    height = np.loadtxt(SOURCE, delimiter=",", dtype=np.float64)
    if height.shape != (701, 701):
        raise RuntimeError(f"FORMAL_TERRAIN_SHAPE_MISMATCH: {height.shape}")
    # Source CSVs are stored in declared ``xy`` order and are transposed by
    # HeightmapIO into the runtime's row-y/column-x convention.
    stored_x, stored_y = np.indices(height.shape, dtype=np.float64)
    x_m = stored_x * 0.05
    y_m = stored_y * 0.05
    # The dedicated loose-ore approach layer continues across the heightmap's
    # x=0 interface onto the separately modelled support apron.  The production
    # pile itself remains closed; this explicit acceptance pad is non-zero at
    # the boundary because the staged real tracks overlap only the first grid
    # column before reversing.  The transverse envelope covers both tracks.
    longitudinal = np.zeros_like(height)
    active_x = (x_m >= 0.0) & (x_m < 5.65)
    longitudinal[active_x] = (
        0.55 + 0.45 * np.sin(np.pi * x_m[active_x] / 5.65) ** 2
    )
    transverse = np.exp(-0.5 * ((y_m - 17.5) / 3.0) ** 8)
    # Keep the support layer shallow enough that it exercises TrackSoil
    # without materially changing the audited bucket/pile cutting geometry.
    # At the staged x=0 boundary this is 0.55 * 0.04 = 0.022 m.
    loose_ore_toe = 0.04 * longitudinal * transverse
    result = np.maximum(height, loose_ore_toe)
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise RuntimeError("TRACK_PILE_ACCEPTANCE_TERRAIN_INVALID")
    np.savetxt(OUTPUT, result, delimiter=",", fmt="%.9f")
    added_volume_m3 = float(np.sum(result - height) * 0.05 * 0.05)
    print({
        "output": str(OUTPUT),
        "shape": result.shape,
        "resolution_m": 0.05,
        "maximum_added_height_m": float(np.max(result - height)),
        "added_volume_m3_cell_area_approximation": added_volume_m3,
        "parameter_status": "ENGINEERING_ACCEPTANCE_SCENARIO_NOT_SITE_CALIBRATED",
    })


if __name__ == "__main__":
    main()
