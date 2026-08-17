# Physics Core V3 Flow / Arrest Causal Audit

This audit uses the existing pre-fix production run
`run_1786528807`, one explicit early-state field export from that run, source
code which defines the scalar diagnostics, and exactly one post-fix production
replay. No material, friction, start/stop angle, damping, velocity, deposition,
resolution, trajectory, or physics timestep parameter was changed.

## Root cause

`FLOW_ARREST_ROOT_CAUSE = MULTIPLE_WITH_BREAKDOWN`

1. **YIELD_RETRIGGER_OWNERSHIP_BUG (primary).** The 2237 reported cells came
   from `DeviceLargeAvalancheBridge.initialize`: central differences are taken
   on `H_free = H_resting + h_mobile`, the eligible Resting layer is
   `min(H_resting, 0.08 m)`, and `unstable=1` exactly when the cohesive
   `Y_start = tau_drive - tau_resist_start` is positive. This is not a legacy
   fixed-angle detector. A tranche latch was cleared only when `Y_stop<=0` and
   local speed was quiet. Consequently, a still-yielded Resting face from which
   Mobile had already transported away remained latched forever and was
   excluded from mobilizable volume.
2. **SETTLED_DETECTOR_PHYSICS_MISMATCH (secondary).** The same physical
   `LOCAL_STATIC_INSTABILITY` component was also used to seed the fixed-angle
   MiniSlope frontier. The existing run therefore conflated physical Y_start,
   candidate residual tiles, and residual ownership. At the explicit early
   post-dig field, only 124 cells / 0.310 m2 satisfy cohesive Y_start, while the
   old stop-angle edge detector flags 190642 cells / 476.605 m2. Flattening the
   latter would alter large cohesive-stable pile faces merely to satisfy a
   geometric repose detector.

There is no evidence that the pre-fix 2237 cells were stale or cohesive-stable.
There is also insufficient final-field evidence to label the remaining
`0.00017983856 m3` Mobile reservoir as a numerical tail: the old run saved only
scalar telemetry at 146.5 s and did not save final height, momentum, speed, or
flux arrays.

## Pre-fix final mask attribution

The pre-fix run did not persist cell indices at its final observation. Exact
NPZ masks, per-mask component counts, and per-cell Mobile distributions cannot
be reconstructed after the DEVICE process has exited. They are not fabricated
here. What can be proved from the detector definition and saved scalars is:

| Mask | Final result at 146.5 s |
|---|---|
| `physical_yield_start_mask` | 2237 cells, 5.5925 m2; largest component 2162 cells / 5.405 m2; component count and eligible volume were not persisted |
| `physical_yield_stop_mask` | not persisted |
| `large_avalanche_candidate_mask` | empty under the pre-fix gated candidate definition because classification was `LOCAL_STATIC_INSTABILITY`; the physical Y_start mask itself was not empty |
| `mobile_dynamic_mask` | not persisted; only total Mobile volume was saved |
| `residual_violation_mask` | not independently evaluated; it was conflated with the physical instability seed |
| `legacy_or_geometric_instability_mask` | not persisted |
| `residual_pending_mask` | four active tiles; tile IDs/cell mask were not persisted in this run |

Because `reported_unstable_mask == physical_yield_start_mask` by construction,
the mutually exclusive A--E assignment of all 2237 reported cells is exact:

| Category | cells | area (m2) | associated volume |
|---|---:|---:|---|
| A_PHYSICALLY_YIELDED_RESTING | 2237 | 5.5925 | not persisted; upper bound from the 0.08 m eligible layer is 0.4474 m3 |
| B_PHYSICALLY_FLOWING_MOBILE | 0 within the reported-mask partition | 0 | 0 |
| C_ARRESTABLE_MOBILE | 0 within the reported-mask partition | 0 | 0 |
| D_RESIDUAL_ONLY | 0 | 0 | 0 |
| E_STALE_OR_INCONSISTENT_MASK | 0 | 0 | 0 |

This A--E partition classifies the 2237 **reported unstable cells**. It does not
claim that no B/C Mobile cells existed elsewhere, because those final Mobile
fields were not saved.

For traceability, exact masks and flux distributions were reconstructed at the
available `EARLY_POST_DIG` snapshot, explicitly marked non-final. There, the
old run had:

- physical Y_start: 124 cells, 7 components, largest 112 cells / 0.280 m2;
- physical Y_stop Mobile: 1108 cells / 2.770 m2;
- Mobile: 4876 cells, 5.238151 m3;
- physically moving Mobile: 0.613084 m3;
- speed p95/max: 0.24038 / 2.20590 m/s;
- outgoing flux p95/max: 0.004781 / 0.065545 m3/s;
- kinetic-energy proxy: 76.35 J;
- integrated momentum norm: 105.85 kg m/s;
- classification: `PHYSICAL_FLOW`.

The corresponding JSON and all masks are in the original run directory as
`v3_flow_arrest_early_post_dig_audit.json` and
`v3_flow_arrest_early_post_dig_masks.npz`.

## Minimal code change

1. CPU and GPU tranche latches now clear when local Mobile has transported
   away, even if the newly exposed Resting face remains above Y_start. If
   Mobile remains locally present, the existing Y_stop hysteresis is retained.
2. The GPU Core now evaluates physical cohesive yield before residual
   projection. Pending frontier tiles cannot own the state while any current
   physical Y_start cell exists.
3. For the cohesive reference material, once physical Y_start is empty, old
   fixed-angle candidate seeds are discarded without modifying H_free. They
   are not treated as proof that a cohesive-stable face must be flattened.
4. `PhysicsDiagnostics.terrain_state` now follows the actual owner. A pending
   frontier is `RESIDUAL_PROJECTION_PENDING`; physical yield/Mobile remains
   `DYNAMIC_MOBILE_FLOW`. The contradictory combination
   `QUASI_STATIC_RESIDUAL_PROJECTION / DEVICE_DYNAMIC_ACTIVITY / 0 iterations`
   is removed.
5. An explicit acceptance-only causal exporter now writes the seven requested
   masks, mutually exclusive A--E masks, component/area/volume statistics,
   h/speed/momentum/Y_start/Y_stop/flux distributions, kinetic proxy, and
   momentum norm at `ARREST_FINAL`. It performs no normal-path transfer and
   does not control physics.

The focused retrigger contract passed (`1 passed`). The existing broader
legacy test file contains unrelated cohesion-based expectation failures and is
not used as closure evidence.

## Exactly one post-fix production replay

Run: `outputs/390f_v2/interactive_runs/run_1786531861`.

It used the same 390F, 701x701 grid, 0.05 m resolution, trajectory, reference
material, timestep, GPU_RUNTIME and DEVICE authority. The replay was externally
terminated with exit code 143; no Python, CUDA, PhysX, or recorded runtime
failure exists. The last persisted state was:

- simulation time: 100.000005 s;
- phase: DEPOSITION;
- Mobile: 2.591029615 m3 (peak 6.188319296 m3);
- payload: approximately zero after a real peak of 0.335270136 m3;
- physical Y_start cells: 2412;
- largest physical component: 1991 cells / 4.9775 m2;
- classification: `PERSISTING_LARGE_UNSTABLE_REGION`;
- actual owner/reason: `DYNAMIC_MOBILE_FLOW / DEVICE_DYNAMIC_MOBILE_FLOW`;
- residual iterations: 0; four candidate tiles pending but not owning;
- maximum absolute mass error: `3.11502e-11 m3`;
- applied soil-force peak: `111384.19 N`.

Earlier in the same post-fix replay, still-yielded terrain re-entered
`LARGE_AVALANCHE_MOBILE_PATH` and transferred another 0.0096 m3 in a physical
step. This demonstrates that the ownership deadlock was removed. It does not
demonstrate final arrest.

`ARREST_FINAL_CAPTURED = NO`.

The post-fix replay did not reach `terrain_settled=true`, did not execute
residual projection, and did not remain stable for the settled-persistence
window before external termination. No second replay was started.

## Remaining physical limitation

Flow/arrest closure is still not accepted. The post-fix run proves that genuine
physical yield now retains ownership, but at its last observation it still had
a macroscopic Mobile reservoir and a significant cohesive Y_start component.
Because the replay ended before final per-cell Y_stop/flux capture, this audit
does not promote that observation to
`TRUE_FLOW_ARREST_MODEL_DEFICIENCY` and does not propose constitutive tuning.
The current status remains:

`FLOW_ARREST_CLOSURE_NOT_YET_DEMONSTRATED`.

