# Packaging audit

Audit date: 2026-08-12. This is a non-destructive extraction. The original
`/home/eric/Desktop/mesh` tree was not moved, renamed or deleted. Imports and
production call sites—not names alone—were inspected.

## Production dependency graph

```text
SoilPhysics facade
  -> EarthmovingPhysicsCore
     -> ContinuousSweep -> ToolTerrainIntersection
     -> FailureSurface V3 -> Resting→Mobile activation
     -> Mobile -> BucketIntake -> Payload
     -> SoilForce (FEE + measured Mobile momentum)
     -> Retention -> Airborne -> Deposition
     -> LargeAvalanche -> Mobile -> residual MiniSlope
     -> TrackSoil (same Resting/Mobile authority)
     -> MassLedger
  -> HOST: BulkStateManager / TerrainState
  -> GPU: DeviceBulkState / GpuRuntimeMetadata
          -> device Failure/Intake/Airborne bridges
          -> Warp Mobile/Track/Deposition/CompactFrontier
  -> IsaacSoilAdapter
     -> dirty H_free tile publication
        -> ChunkedDynamicMeshAdapter (viewport)
        -> IsaacChunkedContactMeshAdapter (optional PhysX recook)
```

`H_free = H_resting + h_mobile` is the shared physics/visual source. No visual
mesh is read back as physics terrain.

## Module-by-module classification

| Classification | Frozen modules/files |
|---|---|
| A REQUIRED_PRODUCTION_SOIL_CORE | `runtime/v2_physics_core.py`, `bulk_state/manager.py`, `bulk_state/models.py`, `bulk_state/volume_integrator.py`, `bulk_state/mass_ledger.py`, `bulk_state/internal_fill.py` |
| B BUCKET_SOIL | `interaction/continuous_sweep.py`, `bulk_interaction/geometry.py`, `bulk_interaction/failure_zone.py`, `bulk_interaction/yield_criterion.py`, `bulk_interaction/model.py`, `bulk_interaction/bucket_intake.py`, `soil_force/model.py`, `tools/bucket_geometry.py`, `tools/tool_descriptor.py`, `tools/tool_descriptor_loader.py`, `tools/tool_kinematics_adapter.py`, `tools/marker_validator.py` |
| C TRACK_SOIL | `bulk_interaction/track_soil.py`, `bulk_interaction/warp_track_soil.py` |
| D GPU_DEVICE_RUNTIME | `runtime/bulk_state_authority.py`, `runtime/gpu_runtime_metadata.py`, `runtime/gpu_bulk_operator_chain.py`, `runtime/gpu_failure_bridge.py`, `runtime/gpu_intake_bridge.py`, `runtime/gpu_airborne_bridge.py`, `runtime/gpu_large_avalanche.py`, `performance/warp_backend.py`, `bulk_interaction/warp_mobile_layer.py`, `bulk_interaction/warp_deposition.py`, `solvers/warp_compact_frontier.py` |
| E ISAAC_ADAPTER | New `earthmoving_soil/isaac/adapter.py`; reused `visualization/dynamic_mesh_adapter.py`, `visualization/chunked_mesh_adapter.py`, `contact/chunked_contact.py`, `contact/isaac_chunked_contact.py`, `soil_force/isaac_adapter.py` |
| F MATERIAL_CONFIG | New `core/config.py`, `configs/materials/iron_ore_reference.yaml`, `configs/solver/default.yaml`, `configs/runtime/gpu_device.yaml`; reused `config/loader.py` |
| G RL_RELEVANT_OUTPUT | New `core/state.py`, `rl/feedback.py`, `diagnostics/physics_diagnostics.py`; reused `runtime/physics_diagnostics.py` |
| H VALIDATION | New lean `tests/test_sdk_alpha.py`; retained production validation history only as documentation, not migrated wholesale |
| I LEGACY | Old `interaction/excavation_operator.py` and duplicate `terrain/{terrain_state,mass_ledger}.py` are excluded. Reference/event-driven MiniSlope adapters remain because the production Core exposes explicit comparison backends; they are not public API |
| J VEHICLE_390F_SPECIFIC | `isaac_loader/run_390f_v2.py`, `runtime/v2_config.py`, all `vehicle/`, `robot/`, `operation/` and 390F configs/assets: excluded |
| K PRESENTATION_ONLY | presentation shell/UI/HUD/camera/payload-visual code: excluded |
| L RESEARCH_ONLY | `audit/`, `dataset/`, `planning/`, `telemetry/`, acceptance/forensic tools and flow-arrest audit helper: excluded |
| M GENERATED_OUTPUT | `outputs/`, interactive runs, checkpoints, NPZ/CSV, videos, screenshots, logs, profiling: excluded |
| N UNKNOWN | None after import audit; optional Isaac bindings remain runtime-provided rather than vendored |

Additional imported support modules retained inside the internal production
namespace are `bulk_exchange/{airborne,dump,retention}.py`,
`bulk_interaction/{mobile_layer,optimized_mobile_layer,deposition,large_avalanche}.py`,
`performance/{active_domain,profiler}.py`, and
`solvers/{base_solver,sparse_tile_slope}.py`. Their package `__init__.py` files
are included. Presentation lighting/debug helpers and legacy wheel/contact
backends are excluded; only production dirty visual/contact chunks remain.

## Track input audit

Production TrackSoil consumes footprint masks, left/right effective surface
velocity, base velocity and `dt`. It does not consume rigid pose or angular
velocity directly; the facade uses pose only to rasterize a footprint. It does
not consume sprocket angular speed, explicit longitudinal slip or lateral slip.
Optional belt speed is converted by the adapter to effective surface velocity,
which is an already-existing production input. No constitutive equation changed.

## Asset and secret audit boundary

No 390F USD, vehicle binary, desktop asset, video, screenshot, forensic NPZ,
token, key, `.env`, credential or company-specific controller is included.
The reference bucket in examples is parameterized and non-proprietary.
