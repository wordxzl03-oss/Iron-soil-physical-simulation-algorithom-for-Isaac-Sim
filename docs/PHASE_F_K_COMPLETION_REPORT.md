# Phase F–K implementation and acceptance report

Date: 2026-08-07  
Scope: additive development on the frozen P0 and completed Phase B–E foundation

Update 2026-08-10: Phase F bucket intake now consumes a single authoritative
`BucketGeometryDescriptor` (L1 explicit profile or validated markers), uses the
bottom plate for separation input, and integrates conservative 2.5-D flux over
the projected mouth segment. The old sampling band is removed. Focused analytic,
orientation, conservation and spatial/time convergence evidence is in
`outputs/bucket_geometry_acceptance.json`; this does not alter the Isaac runtime
gate or upgrade the report's broader sim-to-real claims.

Update 2026-08-10 (390F SoilForce): the current machine integration target is
the tracked hydraulic 390F excavator, not the WheelLoader. Its CAD-derived
bucket descriptor, momentum-based force refinement and external literature
benchmark are documented in `REAL_390F_BUCKET_INTEGRATION.md`,
`SOIL_FORCE_THEORY_TRACE.md` and `SOIL_FORCE_LITERATURE_BENCHMARKS.md`.
The older WheelLoader P0 remains only a frozen regression and is not evidence
for the 390F runtime.

## Outcome

The repository now contains the complete pure-Python mechanism chain for Phases F–K and one shared authoritative material coordinator:

```text
Tool sweep candidate
  -> material-dependent failure wedge
  -> resting-to-mobile activation
  -> conservative local mobile transport
  -> bucket-mouth relative flux / bounded payload
  -> geometry/effective-gravity retention
  -> coarse ballistic parcels
  -> deposition
  -> event-driven MiniSlope relaxation
  -> shared-wedge soil force
  -> VehicleCommand-only operation state machine
  -> target-driven attack/path/tracking baseline
  -> synchronized episode dataset
```

The pure suite and CPU scalability matrix pass. Final 390F Isaac-integrated
acceptance is not claimed because this pure acceptance does not execute a fresh
closed-loop SimulationApp episode. This does not invalidate the saved earlier
runtime probes or the new pure numerical and literature-comparison evidence.

## Phase status

| Phase | Implemented | Pure evidence | Isaac evidence |
|---|---|---|---|
| F | intersection semantics, unified Luengo-based critical wedge/activation geometry, conservative mobile layer, mouth flux, deposition, MiniSlope hook | PASS | PENDING |
| G | geometry/effective-gravity retention, exact parcel split, ballistic landing, terrain/receiver/export dump, post-dump MiniSlope | PASS | PENDING |
| H | shared-wedge FEE-inspired force, public `RigidPrim` force adapter, action-event payload MassAPI adapter | PASS | BLOCKED |
| I | all 14 states, observation-driven transitions, shared `VehicleCommand`, F/G/H cycle coordinator, no pose writer | PASS | BLOCKED |
| J | four goal types, current-pile attack generation, Pareto/weighted evaluation, articulated state lattice, smoothing, bounded nonlinear tracker, safety monitor, per-action replanning | PASS | BLOCKED |
| K | atomic NPZ+JSON episode schema, SHA-256, synchronized arrays, 12-case scalability, parameter sensitivity, debug-layer schema | PASS | BLOCKED |

`PASS` in the pure column means mechanism/unit acceptance only. It is not a substitute for an Isaac/PhysX run.

## Key contracts

- `H_resting_m` remains the authoritative vertex field with `H[y,x]` and `triangle_a_c` volume integration.
- Visual, contact, planner, and debug representations are derived views.
- Candidate swept volume is never payload and never directly deletes terrain.
- All resting/mobile/payload/airborne/outflow changes use explicit `ConservativeTransfer` records.
- `estimated_payload_mass_kg` always means assumed-density mass; `true_mass` is rejected by the episode schema.
- MiniSlope sees only resting terrain and must report its boundary outflow consistently with measured volume change.
- Failure-zone analytical geometry, conservative raster activation and soil-force calculations share the same `FailureStripGeometry`; see `FAILURE_ZONE_THEORY_TRACE.md`.
- Failure separation input, mouth flux, payload capacity and retention profile now derive from the same `BucketGeometryDescriptor`; see `BUCKET_GEOMETRY_THEORY_TRACE.md`.
- Normal operation and planning output `VehicleCommand`; no root pose setter exists in the new operation/coordinator source.
- Planning terminates against mass, volume, remaining ratio, or target heightmap—not an assumed scoop count.
- P0 `interactive_dig_demo.py` remains unchanged as the fixed six-scoop regression.

## Reproduce current evidence

```bash
cd /home/eric/Desktop/mesh
.venv/bin/python -m pytest -q tests
.venv/bin/python tools/phase_a_baseline.py validate
.venv/bin/python tools/run_phase_k_benchmark.py --repeats 3
.venv/bin/python tools/run_phase_f_to_k_acceptance.py
.venv/bin/python tools/run_failure_zone_acceptance.py
.venv/bin/python tools/run_bucket_geometry_acceptance.py
```

Expected at this revision:

- full modular suite: PASS;
- frozen P0 baseline: PASS;
- Phase-K matrix: 12/12 PASS for 128/256/512/701 × small/medium/large;
- F–K pure acceptance: PASS;
- F–K overall status: BLOCKED because the required new Isaac runtime evidence does not exist yet.

Evidence:

- `outputs/phase_f_to_k_acceptance.json`
- `outputs/failure_zone_acceptance.json`
- `outputs/bucket_geometry_acceptance.json`
- `outputs/phase_k_scalability_benchmark.json`
- `outputs/phase_bcde_acceptance_validation.json`
- `outputs/phase_a_baseline_manifest.json`

## Required runtime closure after the platform gate clears

Run fresh processes; do not reuse stale stages:

```bash
./run_phase_cd_acceptance.sh --/log/level=error
.venv/bin/python tools/validate_phase_bcde_acceptance.py
```

Then execute the integrated H/I/J acceptance entrypoint against the audited
390F USD. Required evidence is:

1. soil force applied at the bucket through the public force API and a measurable speed/load/energy difference against force-disabled control;
2. one continuous drive–dig–reverse–transport–dump–return cycle with zero normal-loop pose writes;
3. visual payload/spill/parcels/deposition synchronized to the authoritative reservoirs;
4. articulated path tracking and action-level replanning until the configured target terminates;
5. one complete `isaac-bulk-episode/v1` record from that same run;
6. H0/drive/dig/post-dig/dump screenshots with readable lighting.

Until these exist, World Model/RL readiness remains `BLOCKED`, not `PASS`.
