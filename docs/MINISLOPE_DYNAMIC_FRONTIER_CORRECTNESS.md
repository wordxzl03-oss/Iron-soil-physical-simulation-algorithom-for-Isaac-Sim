# MiniSlope dynamic-frontier correctness gate

Date: 2026-08-10

## Outcome

The fixed local crop in the EventDriven CPU correctness path has been removed.  A
terrain edit is now only an instability seed.  The optimized backend executes
the same four legacy neighbour directions and two checkerboard phases as the
full-grid CPU reference while maintaining a monotone, dynamically expanding
reachable cell frontier.

After every conservative edge transfer, both endpoints join the frontier.
All adjacent slopes are consequently reconsidered in later vector phases and
the active tile envelope can expand across any number of tile boundaries.  No
runout estimate, fixed safety margin, bounding box, active-area cap, or Python
single-cell queue determines the final propagation range.

`SolverConfig.max_iterations` remains visible as the sustained-avalanche
classification threshold in this optimized backend, but it is not used to
truncate a reachable avalanche.  A solve which crosses that threshold or
reaches at least half of the terrain is classified as
`LARGE_SUSTAINED_AVALANCHE`; diagnostics reserve the future Resting-to-Mobile
transfer hook without changing the current quasi-static MiniSlope physics.

This `EventDrivenMinimumSlopeAdapter` still runs each vector phase over the
overall `reached_bounds` bounding rectangle.  Its tile array is diagnostic and
does not drive numerical work.  It is therefore retained and labelled
`CPU_OPTIMIZED_REACHABLE_BBOX_BASELINE`; it is not claimed to be a true sparse
tile-frontier implementation.

`MinimumSlopeAdapter` is now strictly the full-domain CPU reference.  It no
longer applies the former precomputed local crop.

## Acceptance

The dedicated acceptance module is
`tests/test_minislope_dynamic_frontier_acceptance.py` and covers:

1. Small local collapse: optimized final height and volume equal the full-grid
   reference.
2. Toe excavation: instability travels beyond its seed footprint and across
   multiple 8-cell tiles, with the final field equal to the full-grid result.
3. Whole-pile stress: the event deliberately requires more quasi-static rounds
   than the configured threshold, reaches the whole 33 x 33 test pile, is not
   truncated, and matches the reference in conserved volume, final heightmap,
   edge-slope quantiles/histogram, and changed-region extent.

Run with:

```bash
cd /home/eric/Desktop/mesh
PYTHONPATH=.:src .venv/bin/pytest -q \
  tests/test_minislope_dynamic_frontier_acceptance.py
```

## Preserved scope

The previously completed Mobile Layer and general active-domain files were not
reverted or replaced.  GUI, GPU/Warp, and later V2 feature work was held while
this correctness gate was implemented and tested.

## Subsequent compact-tile backend

The later performance backend is
`SparseTileFrontierMinimumSlopeAdapter`. It consumes compact tile IDs and
unique compact edge coordinate buffers. Stable tiles leave the next frontier;
transfers batch-activate the tiles owning every adjacent endpoint stencil.
`CompactActiveEdgeBatch.kernel_buffers()` is the data boundary reserved for a
future Warp/GPU kernel. See `MINISLOPE_SPARSE_TILE_PERFORMANCE.md` for its
701 x 701 measurements and limitations.
