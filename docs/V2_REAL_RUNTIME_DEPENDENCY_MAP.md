# V2 real runtime dependency map

This is the production-path map used for the final GPU integration. It is a
strict snapshot of current call sites, not a claim that the interactive GPU
runtime is enabled.

| Real runtime component | Authority | Terrain input | Output | Full-field read in normal path | Compact alternative available | Integration status |
|---|---|---|---|---:|---:|---|
| `DeviceBulkState` | DEVICE | resident fields | shared Warp fields, scalar ledger, dirty tiles | no | n/a | `DEVICE_NATIVE` |
| Warp TrackSoil / Mobile / Deposition | DEVICE | DeviceBulkState | DeviceBulkState | no | n/a | `DEVICE_NATIVE` |
| `DeviceFailureZoneBridge` | DEVICE + compact CPU geometry | sweep-derived physical patch | Device activation/momentum/forcing scatter | no | yes | `DEVICE_AUTHORITY_COMPACT_HOST_BRIDGE`, not called by Core yet |
| `DeviceBucketIntakeBridge` | DEVICE + compact CPU flux | mouth-control-volume patch | Device Mobile removal + payload transaction | no | yes | `DEVICE_AUTHORITY_COMPACT_HOST_BRIDGE`, not called by Core yet |
| `DeviceAirborneBridge` | DEVICE + host sparse parcels | batched bilinear device query | Device landing scatter | no | yes | `DEVICE_AUTHORITY_COMPACT_HOST_BRIDGE`, not called by Core yet |
| `EarthmovingPhysicsCore` | HOST | `BulkStateManager.current_state` fields | `TerrainState` | yes | bridge chain exists | `INVALID_FULL_FIELD_HOST_DEPENDENCY` |
| `run_390f_v2.py:current_surface_height` | HOST | `physics_core.state` Resting + Mobile | track contact / visual | yes | dirty tiles | `INVALID_FULL_FIELD_HOST_DEPENDENCY` |
| `run_390f_v2.py:operation_observation` | HOST | Resting/Mobile integration | state-machine conditions | yes | device scalar/point query | `INVALID_FULL_FIELD_HOST_DEPENDENCY` |
| `run_390f_v2.py` pre-dig intersection | HOST | full Resting field | UI penetration/intersection | yes | Core compact result | `INVALID_FULL_FIELD_HOST_DEPENDENCY` |
| `run_390f_v2.py` failure/logging | HOST | full state | failure JSON | yes | explicit acceptance snapshot only | `INVALID_FULL_FIELD_HOST_DEPENDENCY` |
| LargeAvalanche controller | HOST | Resting/Mobile/Momentum full fields | physical transition | yes | active-region/device-assisted representation pending | `NOT_YET_CONNECTED` |
| PhysX contact chunks | PHYSX contact view | host chunk samples | normal support | currently full host source | dirty-tile chunks | `NOT_YET_CONNECTED` |
| track traction controller | HOST scalar / PhysX wrench | actual body velocity/contact | bounded track forces | no terrain full read itself | compact footprint / scalar output | `HOST_SCALAR_ONLY`; closed-loop evidence pending |

## Required migration order

1. Make the existing `EarthmovingPhysicsCore` backend-aware without retaining
   a host terrain authority in GPU mode; expose only scalar snapshot, compact
   point query, patch and dirty-tile interfaces.
2. Replace launcher `current_surface_height`, state-machine terrain queries,
   pre-dig diagnostic intersection, and normal logging with those interfaces.
3. Bind the existing compact FailureZone, intake and airborne bridges to that
   Core, then route TrackSoil/Mobile/Deposition through its one shared chain.
4. Move LargeAvalanche to a device-assisted active region before GPU mode is
   selectable.
5. Publish dirty chunks to PhysX and demonstrate that later true track
   contacts observe earlier device deformation before running the one-cycle
   acceptance.
