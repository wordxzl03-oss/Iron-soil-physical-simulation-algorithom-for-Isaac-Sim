# V2 GPU-authoritative runtime baseline

## Status

`GPU_STATE_OWNERSHIP_STATUS = IN_PROGRESS`.

The project must not claim an integrated GPU-authoritative earthmoving cycle
yet. The interactive launcher remains explicitly `HOST_REFERENCE`; selecting
`GPU_RUNTIME` is rejected rather than silently falling back to the legacy NumPy
core.

## Ownership audit

Before this baseline, `BulkStateManager` was the sole owner of immutable NumPy
`TerrainState`, while each Warp operator allocated a separate device copy. That
architecture was unsuitable for a GPU runtime because it could produce three
device shadows plus a host authority.

The new state boundary is:

```text
HOST_REFERENCE
  BulkStateManager / TerrainState (host authority)

GPU_RUNTIME (being integrated)
  DeviceBulkState (one Warp authority)
    ├─ resting, mobile, momentum_x/y, Triangle-A-C weights
    ├─ active/material/dirty masks
    ├─ TrackSoil rut field
    ├─ MiniSlope frontier/reached/tile/edge state
    ├─ avalanche slope/unstable scratch
    └─ deposition work/mask
       ↓ shared bindings, no shadow terrain
  WarpMobileLayerSolver
  WarpTrackSoilOperator
  WarpCompactActiveEdgeOperator
  WarpDepositionOperator
       ↓
  HostBulkStateView / HostBulkStatePatch
  (read-only explicit debug, acceptance, checkpoint, visual/contact or
   compact legacy-geometry bridge only)
```

`HostBulkStateView.commit()` always raises. A device reduction, rather than a
host terrain copy, drives `DeviceMaterialLedger` reservoir checks.

## Current implemented invariants

- Device weights now exactly match `TerrainVolumeIntegrator` Triangle-A-C
  topology, not a trapezoidal approximation.
- Full-field H2D/D2H counts are reset per normal physics step and exposed in
  `DeviceTransferSnapshot`.
- Full host views are only permitted for `debug`, `acceptance`, `checkpoint`,
  `reset`, or `visualization` boundaries.
- Dirty visual/contact data can be extracted tile-by-tile. The normal compact
  bridge transfers tile samples, not an entire 701×701 state.
- Dirty tiles are cell-aligned with visual/contact chunks and include their
  final shared vertex row/column. `ChunkedDynamicMeshAdapter` and
  `IsaacChunkedContactMeshAdapter` now accept these compact samples directly;
  they do not require a temporary full-heightmap snapshot.
- TrackSoil has a compact-index interface; the GPU runtime path uploads active
  footprint indices instead of a full terrain mask.
- Reset restores all mutable device fields, masks, scratch buffers and the
  reset generation.

## Controlled integrated device window

`tools/run_gpu_device_window_acceptance.py` executes a real CUDA 701×701,
0.05 m synchronized window containing `TrackSoil → Mobile → Deposition` on
one `DeviceBulkState`, then compares its explicit acceptance snapshot with the
same host-reference operators. The generated
`outputs/390f_v2/gpu_device_window_acceptance.json` is a **subsystem** result,
not a one-cycle result.

Latest measured acceptance:

- `GPU_DEVICE_WINDOW_STATUS = PASS`
- Resting L∞ error: `0.0 m`; Mobile L∞: about `1.7e-17 m`; momentum L∞:
  about `7.3e-18 m²/s`.
- Track/deposition/mobile-volume differences are below `3e-16 m³`.
- Normal physics transfer accounting reports full-field H2D = `0` and
  full-field D2H = `0`. The only normal host traffic is compact track indices,
  scalar reductions and the compact dirty-tile flags.
- The exercised event touched two dirty tiles for TrackSoil, two for Mobile,
  and none for deposition; dirty tile publication is not a global refresh.

The acceptance script has explicit numerical tolerances and exits non-zero on
an equivalence or full-transfer violation. It must not be relabelled as
`INTERACTIVE_RUNTIME_GPU_RESIDENT` until the real core and launcher complete
the remaining migration below.

## Measured LargeAvalanche CPU hotspot boundary

`LargeAvalancheTransitionController.last_profile_ms` now separately records
host `gradient`, `connected_components`, `diagnostics`, `transfer`, and
`total` costs. `EarthmovingPhysicsCore` emits these per-step values as
`large_avalanche_*` timing fields. This establishes an Amdahl measurement
boundary before any GPU rewrite of the accepted transition semantics.

## Newly available compact physics bridges

The following bridges preserve the accepted CPU geometry/parcel formulas but
remove their requirement for a complete host terrain field:

- `DeviceFailureZoneBridge`: a real CAD sweep requests a patch based on the
  implemented physical maximum wedge length. It dynamically expands if the
  solved wedge reaches the local boundary, then scatters conservative
  Resting→Mobile activation, momentum and compact forcing indices directly to
  DeviceBulkState.
- `DeviceBucketIntakeBridge`: derives its patch from the exact mouth
  control-volume bounds already used by `BucketIntakeModel`, computes the
  existing flux/COM update locally, and scatters only changed Mobile/momentum
  cells. Its `DevicePayloadTransaction` rejects a payload mismatch.
- `DeviceAirborneBridge`: keeps the sparse parcel list on CPU, queries all
  candidate landing heights in one GPU batch, and atomically scatters landed
  Mobile/momentum sources. It never downloads a terrain field.

Both new controlled CUDA 701×701 windows pass with normal full-field transfer
counts of zero:

- [excavation window](/home/eric/Desktop/mesh/outputs/390f_v2/gpu_excavation_window_acceptance.json): real CAD FailureZone → activation → intake;
  FailureZone patch 29,568 cells (6.02% of terrain), mouth patch 177 cells,
  height/momentum errors below `3.6e-15`.
- [dump window](/home/eric/Desktop/mesh/outputs/390f_v2/gpu_dump_window_acceptance.json):
  Payload → Airborne → batched device landing query → incremental deposition;
  all reported field and volume errors are zero for the controlled case.

The detailed before/after call-site classification is in
[GPU_REMAINING_HOST_DEPENDENCY_AUDIT.md](/home/eric/Desktop/mesh/docs/GPU_REMAINING_HOST_DEPENDENCY_AUDIT.md).

## Not yet enabled

The bridge implementations above are not yet selected by the real
`EarthmovingPhysicsCore` / Isaac launcher. The core still owns host full-field
FailureZone, intake, airborne and LargeAvalanche flows, while the launcher
still has normal-path full-terrain observations. Therefore `GPU_RUNTIME`
remains rejected: there is deliberately no one-cycle GPU acceptance result and
no three-cycle run has been started.

This document supersedes any implication that standalone Warp operator passes
mean `INTERACTIVE_RUNTIME_GPU_RESIDENT = PASS`.
