# GPU runtime remaining-host-dependency audit

This audit distinguishes code that is now available to the GPU runtime from
the still-host-authoritative V2 core/launcher path. It is intentionally not a
claim that `GPU_RUNTIME` is selectable yet.

## Dependency graph

```text
DeviceBulkState                         DEVICE_NATIVE
 ├─ WarpTrackSoilOperator               DEVICE_NATIVE
 ├─ WarpMobileLayerSolver               DEVICE_NATIVE
 ├─ WarpDepositionOperator              DEVICE_NATIVE
 ├─ WarpCompactActiveEdgeOperator       DEVICE_NATIVE primitive
 ├─ DeviceFailureZoneBridge             COMPACT_HOST_BRIDGE
 │    CAD sweep → physical patch → CPU FailureZone → device scatter
 ├─ DeviceBucketIntakeBridge            COMPACT_HOST_BRIDGE
 │    mouth control volumes → local patch → CPU intake → device scatter
 ├─ DeviceAirborneBridge                COMPACT_HOST_BRIDGE
 │    CPU parcel list → batched device height query → device scatter
 └─ dirty-tile visual/contact samples   COMPACT_HOST_BRIDGE

EarthmovingPhysicsCore / run_390f_v2.py HOST_REFERENCE / NOT_INTEGRATED
 ├─ BulkStateManager TerrainState       FULL_FIELD_HOST_DEPENDENCY
 ├─ FailureZone / intake / airborne     FULL_FIELD_HOST_DEPENDENCY
 ├─ LargeAvalanche diagnostics          FULL_FIELD_HOST_DEPENDENCY
 └─ visual/contact/state observations   FULL_FIELD_HOST_DEPENDENCY
```

## Normal-path dependency inventory

| Consumer / call site | Previous terrain input | GPU bridge | normal transfer | Status |
|---|---:|---|---:|---|
| `ToolTerrainIntersectionModel` + `FailureZoneModel` | 701×701 resting | geometry-derived patch, physical `maximum_wedge_length_m` expansion | patch D2H + patch H2D scatter | `COMPACT_HOST_BRIDGE_AVAILABLE` |
| Failure activation | full `TerrainState` rebuild | resting/mobile/momentum patch scatter + compact forcing indices | local only | `DEVICE_COMMIT_AVAILABLE` |
| `BucketIntakeModel` | full mobile/momentum/resting | exact projected mouth-control-volume bbox | local only | `COMPACT_HOST_BRIDGE_AVAILABLE` |
| `AirborneParcelModel.advance` | full resting+mobile surface | batched bilinear device query + compact atomic landing scatter | O(parcel count) | `COMPACT_HOST_BRIDGE_AVAILABLE` |
| Mobile / TrackSoil / deposition | distinct CPU fields | shared DeviceBulkState | no full field | `DEVICE_NATIVE` |
| LargeAvalanche controller | full resting/mobile/momentum NumPy fields | no compact implementation yet; host timing is instrumented | unresolved | `FULL_FIELD_HOST_DEPENDENCY` |
| `EarthmovingPhysicsCore` dig/dump path | `BulkStateManager.current_state` | not switched to bridge chain | full field | `NOT_INTEGRATED` |
| `run_390f_v2.py` `current_surface_height`, contact, visual, observations | `physics_core.state` | dirty publication API exists but launcher still reads host state | full field | `NOT_INTEGRATED` |

## Transfer classifications

- `REQUIRED_SCALAR`: device ledger reductions, phase/state scalars.
- `REQUIRED_COMPACT_PATCH`: FailureZone and mouth geometry patches. Their
  bbox derives from actual CAD sweep/mouth geometry; no fixed 64-cell ROI is
  used. The FailureZone bridge expands when the solved wedge reaches a patch
  boundary.
- `REQUIRED_COMPACT_INDEX`: activated tool-forcing cells, track footprint
  cells, and ballistic landing cells.
- `DEBUG_ONLY_FULL_VIEW` / `ACCEPTANCE_ONLY_FULL_VIEW`: explicit
  `HostBulkStateView`; never allowed in a normal GPU step.
- `INVALID_RUNTIME_FULL_FIELD_READ`: the remaining V2 core/launcher paths
  above. `GPU_RUNTIME` remains rejected while any of them is active.

## Measured controlled bridges

| Window | Result | Full H2D/D2H during normal window |
|---|---|---|
| TrackSoil → Mobile → Deposition | PASS | 0 / 0 |
| real CAD FailureZone → activation → bucket intake | PASS | 0 / 0 |
| Payload → Airborne → batched terrain query → deposition | PASS | 0 / 0 |

The acceptance JSON files are under `outputs/390f_v2/`. These are controlled
numeric-equivalence windows, not a real 390F one-cycle acceptance.
