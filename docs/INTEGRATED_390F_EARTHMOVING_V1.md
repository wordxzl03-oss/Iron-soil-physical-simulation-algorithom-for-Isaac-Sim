# 390F Integrated Earthmoving Physics V1

## Outcome

The repository now contains a completed three-case, three-cycle formal run on
the real 390F USD articulation. The result is an integrated reduced-order
research simulator with conservative material transfer and dynamic
soil-machine feedback. It is **not** accepted as a complete physics V1 and is
not a field-calibrated iron-ore digital twin.

Machine-readable status:

```text
overall = DEGRADED_NOT_FULL_V1_ACCEPTANCE
PHYSICS_MODEL_STATUS = DEGRADED_REAL_BUCKET_RAKE_OUTSIDE_IMPLEMENTED_FEE_DOMAIN
ISAAC_RUNTIME_STATUS = DEGRADED_STATIC_TRACK_SUPPORT_AND_NO_CONTACT_FORCE_SENSOR
LITERATURE_VALIDATION_STATUS = EXECUTED_2D_BENCHMARK_FINITE_WIDTH_AND_CAVITY_BLOCKED
PERFORMANCE_STATUS = FAIL_NOT_REAL_TIME
```

The authoritative result is
`outputs/integrated_390f_v1/integrated_390f_physics_acceptance.json`.

## Integrated path

The executed path is:

```text
390F articulation state
  -> CAD bucket state and swept intersection
  -> applicability-aware failure-zone evaluation
  -> resting/mobile transfer or force-free geometric sweep outside FEE domain
  -> mobile transport and conservative bucket intake
  -> exact bucket internal-fill free-surface solve
  -> fill-dependent secondary separation
  -> retention, spill, airborne transfer and deposition
  -> MiniSlope relaxation
  -> heightmap and visual mesh update
  -> soil wrench and bounded articulation command
  -> next PhysX step
```

`BucketInternalFillState` carries volume, mass, geometric and rated fill ratios,
free-surface normal and offset, occupied cross-section, COM, second moments,
inertia, phase and the secondary separation geometry. The free-surface offset
is solved by half-plane clipping and bisection to match payload volume; it is
not forced through the bucket lip.

Material leaving the bucket is never deleted. Retention and spill transfer it
to airborne state, deposition transfers it to the terrain/mobile ledger, and
MiniSlope runs after each cycle's deposition phase. Payload COM and separation
geometry are recomputed from the remaining fill.

## Formal run contract

| Item | Formal value |
|---|---:|
| Isaac asset | `/home/eric/桌面/bulldozer_sim/bulldozer_main.usd` |
| Cases | A no force; B quasi-static FEE; C full model |
| Cycles per case | 3, with no terrain, payload or vehicle reset |
| Phases per cycle | 12 |
| Physics steps per case | 648 |
| Physics step | 1/60 s |
| Heightmap | 701 x 701 |
| Resolution | 0.05 m |
| Physical grid extent | 35.05 m x 35.05 m |
| Deterministic seed | 390 |
| MiniSlope calls | 3 per case |

The legacy dataset directory contains `25m` in its name, but the formal grid is
701 cells at 0.05 m and therefore spans 35.05 m. No silent resolution downgrade
was used.

All cases use the same machine, initial heightmap, phase policy, controller,
time step, material tuple and seed. Only the applied soil-force mode changes.

## A/B/C results

| Metric | A: no force | B: quasi-static FEE | C: full model |
|---|---:|---:|---:|
| Maximum applied soil force | 0 N | 0 N | 139,082 N |
| Peak payload | 0.03185 m³ | 0.03185 m³ | 0.03728 m³ |
| Final payload | 0 m³ | 0 m³ | 0.03728 m³ |
| Absolute mechanical work | 6.750 MJ | 6.750 MJ | 6.849 MJ |
| Joint RMSE versus A | — | 0 rad | 0.07357 rad |
| Bucket-tip RMSE versus A | — | 0 m | 0.2534 m |
| Final volume-balance error | 9.09e-13 m³ | 9.09e-13 m³ | 2.73e-12 m³ |
| RTF | 0.01373 | 0.01366 | 0.01725 |

Case C proves causal feedback in this run: its applied force equals the full
computed force, is non-zero, and changes both the joint and bucket-tip
trajectories relative to A. The shared positive-power command never exceeds
391 kW.

The B result must not be misread as a successful quasi-static force
calibration. Every recorded real-bucket cutting state was either outside the
implemented Luengo/FEE closure, reverse separation, or no intersection. The
model therefore correctly returned zero quasi-static force instead of
inventing a wedge. Outside the FEE force domain, CAD sweep geometry still
transfers material conservatively but is explicitly marked
`GEOMETRIC_SWEEP_ONLY_OUTSIDE_FEE_FORCE_DOMAIN` and contributes no FEE force.
Case C's non-zero contribution is consequently the full-model momentum term,
not evidence that the missing real-rake quasi-static closure is solved.

## Conservation and continuity

Each case starts cycle 2 and cycle 3 from the preceding terrain, vehicle and
payload states. The C run finishes with:

- initial total volume: 1351.20114605 m³;
- final resting terrain: 1345.69923266 m³;
- final mobile material: 5.46463752 m³;
- final bucket payload: 0.03727587 m³;
- final balance error: 2.73e-12 m³.

The non-zero final payload is retained state, not lost mass and not a reset.

## Real 390F runtime status

The loaded articulation exposes four revolute DOFs: swing, boom, stick and
bucket. The source `targetVelocity=800` is cleared. The formal cycles do not use
pose teleportation. A causal velocity/acceleration shaper and generalized
effort/shared-power limiter advances force-type PhysX drive targets.

The seven runtime bodies have finite COM and inertia and sum to approximately
94 t. That mass exceeds the manufacturer's published 390F L operating range.
The source USD also contains invalid COM/inertia placeholders; Isaac resolves
finite mesh-derived runtime values. These are audited asset/runtime facts, not
a manufacturer-validated mass distribution. No hydraulic cylinder mapping is
claimed because unambiguous cylinder pin geometry was not recovered.

Track and static support collision APIs are enabled. Dynamic bulk collision is
disabled and the bucket-bulk rigid pair is unavailable while the custom bulk
solver applies its wrench, preventing double counting. The Isaac 4.5 track
contact-force sensor view caused a reproducible native crash and is therefore
reported `DEGRADED`; collision-schema presence is not presented as measured
contact force.

## Terrain and timestamp synchronization

The log records robot, soil-force, terrain, visual-mesh and contact-surface
timestamps. Soil-force samples use the current robot state. The visual terrain
is refreshed every six physics steps, with a measured maximum terrain-to-visual
lag of 0.08333 s.

The PhysX track support remains a static collision surface; it is not a
per-step deforming heightfield. Therefore `terrain_visual_sync` passes its
declared event policy, while `terrain_contact_sync` fails. The simulator does
not claim current-heightmap rigid contact.

## Literature and material status

The equation-complete Obermayr straight-blade benchmark remains executed. Its
eight cases have about 25.14% MARE, with the largest underprediction at greater
depth and 0.30 m width. The present strip formulation is nearly linear in
width and omits explicit lateral failure/side-shear mechanics, explaining why
it cannot reproduce the measured finite-width trend.

The available primary sources did not expose enough complete equations and
inputs to implement the McKyes/Ali, Zhang/Kushwaha or cavity-expansion terms
without guessing. No constant width gain or fabricated cavity coefficient was
added. Finite-width 3D resistance, cavity expansion and a second matched
bucket/excavator benchmark remain explicitly blocked.

The formal material scenario uses one near-matched low-stress I3 iron-ore tuple:
1370 kg/m³ bulk density, 29.8° internal friction, 0.8 kPa cohesion and 34.4°
tool-wall friction. The flow start/stop angles remain engineering inputs. This
is condition-coherent literature input, not site calibration.

## Performance

The C run simulated 10.8 s in 626.1 s wall time. Its mean full-step time was
966 ms, mean soil-solver time 791 ms and mean mobile-layer time 636 ms. Each
MiniSlope call averaged 27.15 s. RTF is far below one, so the current 0.05 m
pipeline is not real time. The resolution was retained to expose this tradeoff.

## Verification

`./run_full_repository_tests.sh` passes 252 tests: 202 current modular tests and
50 legacy root tests. Plain `pytest` is scoped to the active `tests/` directory
and passes 202 tests plus 54 subtests.

## Deliverables

The formal output directory is `outputs/integrated_390f_v1/` and contains:

- `integrated_390f_physics_acceptance.json`;
- `three_cycle_log.csv`;
- `cycle_summary.json`;
- `joint_tip_timeseries.csv`;
- `soil_force_timeseries.csv`;
- `payload_timeseries.csv`;
- `mass_balance_timeseries.csv`;
- `terrain_state_sequence.npz`;
- `before_heightmap_m.csv` and `after_heightmap_m.csv`;
- `runtime_profile.json` and `runtime_manifest.json`;
- `condition_consistent_material_scenarios.json`;
- real bucket, runtime mass/inertia and literature evidence JSON files;
- `three_cycle_full_physics.mp4` and its frame-selection manifest.

The video is a time-compressed state-change rendering from the same formal C
run: 116 selected frames at 10 fps. The 31 GB Replicator stream contained tens
of thousands of near-duplicate frames emitted during solver waits; those raw
duplicates were removed after the selected frames and MP4 were verified.

## Reproduction

Offline tests and aggregation:

```bash
./run_full_repository_tests.sh
.venv/bin/python tools/aggregate_integrated_390f_v1.py
```

Formal Isaac cases use `isaac_loader/integrated_390f_v1.py` with cases A, B and
C; C adds rendering. Exact asset, runner hash, material, grid, seed and case
contract are stored in `runtime_manifest.json`.

## Claim boundary

Supported claim:

> Integrated literature-constrained reduced-order 390F earthmoving simulator
> with dynamic soil-machine coupling, conservative bulk-material transfer,
> real excavator geometry, multi-cycle digging and published soil-force
> benchmarking.

Unsupported claim:

> Field-calibrated iron-ore digital twin or fully accepted 3D earthmoving
> physics model.
