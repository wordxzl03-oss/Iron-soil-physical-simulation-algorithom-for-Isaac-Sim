# Production GPU Core checkpoint

Date: 2026-08-11

Status: `PRODUCTION_EARTHMOVING_CORE_GPU_BRANCH = IMPLEMENTED`

The production branch, connected LargeAvalanche device preprocessor and real
launcher migration are implemented.  `GPU_RUNTIME` can be selected explicitly
for controlled acceptance with `--runtime-backend GPU_RUNTIME`; the shipped
configuration remains `HOST_REFERENCE` until the real closed-loop acceptance
run reaches its physical terminal condition.

## Production call graph

Before:

```text
run_390f_v2.py
  -> EarthmovingPhysicsCore (BulkStateManager / TerrainState only)
     -> CPU FailureZone -> CPU Mobile -> CPU Intake
     -> CPU Airborne -> CPU Mobile -> CPU Deposition
     -> CPU LargeAvalanche -> CPU incremental MiniSlope
  -> launcher reads complete TerrainState for observations/publication
```

After:

```text
run_390f_v2.py
  -> EarthmovingPhysicsCore(runtime_backend)
     HOST_REFERENCE
       -> BulkStateManager / TerrainState (unchanged reference semantics)
     GPU_RUNTIME
       -> DeviceBulkState + GpuRuntimeMetadata
       -> GpuBulkOperatorChain
          -> DeviceFailureZoneBridge (physical compact host geometry patch)
          -> WarpMobileLayerSolver (shared resident fields)
          -> DeviceBucketIntakeBridge (bucket-mouth compact patch)
          -> WarpTrackSoilOperator (compact footprint indices)
          -> DeviceAirborneBridge (sparse host parcels + batched device query)
          -> WarpDepositionOperator (resident Resting/Mobile)
          -> DeviceLargeAvalancheBridge
             (resident connected components + Resting-to-Mobile transfer)
          -> WarpCompactActiveEdgeOperator (dynamic compact tile/edge frontier)
  -> scalar reductions + batched surface queries + dirty-tile publication
```

`PhysicsCoreStepResult.state` is deliberately `None` in GPU runtime.  Any
attempt to access `EarthmovingPhysicsCore.state` there raises
`GPU_RUNTIME_HAS_NO_HOST_TERRAINSTATE`.

## Removed normal-loop complete-field reads

- dig pre-diagnostic: GPU branch uses the compact FailureZone result returned
  by the production Core instead of reading `H_resting_m` before the step;
- state-machine cutting-edge surface observation: batched device bilinear
  sampling;
- track support-height observation: compact footprint-coordinate sampling;
- material ledger and launcher telemetry: device reductions plus
  `GpuRuntimeMetadata` scalar reservoirs;
- airborne domain diagnostics: sparse parcel metadata only;
- visual terrain publication: dirty device tiles only;
- PhysX contact publication: the same dirty tile samples, independently
  buffered for the contact update frequency;
- result/debug telemetry: backend-neutral reservoir observations, not
  `core_result.state`.

Full host snapshots remain restricted to explicit debug, checkpoint,
acceptance, reset, or initialization boundaries.

## Finite-time deposition correction

CPU_REFERENCE and GPU_RUNTIME now evaluate Mobile-to-Resting deposition on
the physical free surface `Resting + Mobile`.  The former substrate-only test
could strand a static Mobile blanket above a steep buried Resting face even
though relabelling the blanket does not alter the surface.  The original GPU
run is retained with `INTERRUPTED_NOT_ACCEPTED.json`; it is not a physics
acceptance result.  CPU/Warp equivalence and exact transfer conservation are
covered by dedicated steep-substrate/mobile-blanket regression tests.

The nominal avalanche parameters remain explicitly uncalibrated.  The launcher
also exposes declared `more_mobile`/`less_mobile` transition sensitivities and
the literature JSON contains the less-mobile endpoint of the existing
start/stop/friction envelope.  These are deterministic engineering sensitivity
cases, never site-calibrated values.

## Remaining normal-path host dependencies

- No complete terrain/mobile field is downloaded in a normal GPU physics step.
  LargeAvalanche uses device-resident slope diagnosis, exact 4/8-connected
  union-find, component reductions and conservative Resting-to-Mobile transfer.
  Only fixed-size diagnostics and dirty-tile flags cross to the host.
- CAD FailureZone geometry and bucket-mouth intake equations execute on
  geometry-derived compact host regions and scatter only changed patches.
  These are intentional compact bridges, not full-field dependencies.
- sparse ballistic parcels remain CPU metadata; landing heights are one
  batched device query and landing is a compact scatter.
- Warp Mobile performs scalar CFL/reduction synchronizations.  These are
  scalar D2H operations, not terrain downloads.

## Acceptance evidence

- `outputs/390f_v2/production_gpu_core_acceptance.json`: PASS;
- `outputs/390f_v2/gpu_large_avalanche_acceptance.json`: PASS against the CPU
  reference for localized, medium and whole-701 cases;
- production Core operator record contains FailureZone, Intake, Mobile,
  TrackSoil, Airborne, Deposition, device LargeAvalanche and Compact Frontier;
- `TerrainState` returned: false;
- normal full-field H2D: 0;
- normal full-field D2H: 0;
- mass-balance error: `-6.821210263296962e-13 m3`;
- Python regression: 258 passed, 11 skipped, 57 subtests passed;
- real Isaac no-soil machine acceptance: PASS.

The interrupted Host reference run is explicitly marked not accepted.  The
GPU real one-cycle run is recorded separately under
`outputs/390f_v2/interactive_runs/` and must reach its physical terminal state
before being reported as a completed cycle.
