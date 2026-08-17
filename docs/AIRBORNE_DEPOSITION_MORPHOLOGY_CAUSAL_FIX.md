# Airborne / Deposition Morphology Causal Fix

## Closure status

`AIRBORNE_DEPOSITION_MORPHOLOGY = PASS`

This change is deliberately narrow. It changes the spatial support of a reduced-order airborne parcel at impact. It does **not** change the 701×701, 0.05 m production grid, physics `dt`, material parameters, Failure Surface V3, Mobile transport, LargeAvalanche/MiniSlope, force-settle behavior, or the accepted conservative ownership contract.

The preserved pre-change source baseline is `deliverables/baselines/pre_airborne_morphology_2026-08-13.tar.gz` (SHA-256 `abe19a74e1404769e4b313600b36118f0d488440d9d7b5f94736bd15bdbf19e4`).

## Causal diagnosis

`MaterialParcel` is a coarse carrier of represented bulk volume, not a microscopic point particle. The old CPU and GPU paths advanced that carrier ballistically, detected impact against `H_free`, and deposited the parcel's entire represented volume at the nearest height-field vertex:

```text
Payload
  -> DumpPhaseOperator release
  -> AirborneParcelModel.create_from_bucket_release
  -> ballistic integration / H_free impact query
  -> DeviceAirborneBridge point scatter
  -> h_mobile + horizontal impact momentum
  -> Mobile-to-Resting deposition
```

The frozen checkpoint contains three parcels of approximately `0.0352766418614 m³` each. An interior grid vertex has a `0.0025 m²` control area, so each atomic write produced

```text
0.0352766418614 / 0.0025 = 14.1106567446 m.
```

The authoritative fields prove that the defect first appears at the landing transaction, not in Failure Surface, Mobile transport, arrest, or the visual mesh:

- `FIRST_BAD_FRAME`: frozen-checkpoint step 82, device time `21.30000000000007 s`; T0 is `21.283333333333406 s`.
- `FIRST_BAD_OPERATOR`: `DeviceAirborneBridge.advance`, write reason `ballistic_parcel_landing`.
- T0 has no deposition delta.
- T1 immediately after the Airborne write has the new delta entirely in `h_mobile` and therefore in `H_free`; `H_resting` is unchanged.
- T2 equals T1 because the production landing write is itself the Airborne-to-Mobile transaction; there is no hidden second conversion.
- T3 has the same `H_free` delta in `H_resting` after Mobile-to-Resting deposition.

The complete T0–T4 arrays, parcel states, momentum and dirty tiles are in `outputs/airborne_deposition_morphology/causal_T0_T4_states.npz`. Operator-boundary metadata is in `causal_T0_T4_metadata.json` and recipient-level evidence is in `parcel_recipient_audit.json`.

## Finite parcel footprint

`AirborneParcelFootprint` carries a compact planar support and its provenance. New release events take the lateral axis from the real release-lip orientation. A legacy checkpoint without footprint metadata is deterministically backfilled from the parcel-release point distribution, with velocity as a single-parcel fallback.

For a real bucket descriptor:

```text
mean bucket depth = effective bucket capacity / bucket mouth area
parcel footprint area = represented parcel volume / mean bucket depth
footprint aspect = cutting-edge length / (mouth area / cutting-edge length)
```

The resulting rectangle is classified as `CONSERVATION_BASED_ENGINEERING_CLOSURE`. It is geometry- and volume-derived, but is not claimed to be a calibrated granular dispersion law. No fixed cell count, Gaussian width, smoothing, clipping, or height cap is used.

At impact, the oriented footprint polygon is clipped against every intersected vertex-centred height-field control region. For overlap area `w_i` and exact topology-aware MassLedger control area `A_i`, including boundary half/quarter areas:

```text
delta_V_i = V_parcel * w_i / sum_j(w_j)
delta_h_i = delta_V_i / A_i
sum_i(A_i * delta_h_i) = V_parcel
```

All weights are non-negative and compactly supported. The same `delta_h_i` initializes the existing Mobile state and scales the existing horizontal impact momentum. Thus `Airborne -> Mobile -> Resting` semantics are preserved.

## Frozen checkpoint before / after

| Quantity | Legacy point scatter | Finite footprint |
|---|---:|---:|
| Initial airborne volume (m³) | 0.105829925584281 | 0.105829925584281 |
| Final airborne volume (m³) | 0 | 0 |
| Deposited volume (m³) | 0.105829925584281 | 0.105829925584281 |
| Unique recipient vertices | 3 | 75 |
| Rasterized footprint area sum (m²) | point support | 0.102916039035236 |
| First-scatter maximum `delta H_free` (m) | 14.110656744571 | 1.028313240362 |
| Final maximum 4-neighbour jump (m) | 14.119127044098 | 0.907103964816 |
| Final anomalous-gradient components | 17 | 3 |
| Largest anomalous-gradient component (cells) | 982 | 40 |
| Mobile nonzero duration in the frozen replay (s) | 0 | 0 |
| Moving-Mobile duration (s) | 0 | 0 |
| Final raw Y-start cells | 49 | 114 |
| Eligible mobilizable volume (m³) | 0 | 0 |
| ARREST_FINAL time after checkpoint (s) | 1.333 (legacy attribution) | 1.367 (causal replay) |

The repaired causal replay reports:

- maximum MassLedger error: `1.1141310096718371e-11 m³`;
- `H_free` net volume error relative to airborne loss: `-7.782663402622347e-14 m³`;
- wall time for 5 simulated seconds: `2.0300472300004913 s`;
- RTF: `2.4629968830817743`.

The difference in raw final Y-start count does not imply an ownership regression: both runs have zero eligible mobilizable volume and settle without retrigger. The repaired footprint simply exposes a different finite spatial boundary than three singular columns.

## Authoritative morphology by transition

| Stage | `H_resting` changed cells | `h_mobile` changed cells | `H_free` changed cells | max positive `delta H_free` (m) |
|---|---:|---:|---:|---:|
| T0 pre-impact | 0 | 0 | 0 | 0 |
| T1 post landing write | 0 | 75 | 75 | 1.028313240362 |
| T2 post A→Mobile | 0 | 75 | 75 | 1.028313240362 |
| T3 post M→R | 75 | 0 | 75 | 1.028313240362 |
| T4 ARREST_FINAL observation | 75 | 0 | 75 | 1.028313240362 |

At T1, `H_free` neighbour jumps are p95 `0.733559308064 m`, p99 `0.886003927808 m`, maximum `0.907103964816 m`; the largest connected anomalous-gradient component contains 40 cells. This remaining footprint edge is reported as a model limitation, not hidden by smoothing.

## Visual mesh A/B

The exact repaired authoritative `H_free` was published in two ways: the current 64-cell dirty-tile update and a full reconstruction. Six dirty tiles covered all 75 changed vertices. The maximum vertex difference was exactly `0.0 m`, so:

`INCREMENTAL_MATCHES_FULL_REBUILD = YES`

The acceptance observer that captured full fields is disabled by default. Production landing uses compact surface samples and indexed writes; the fix adds no normal-step full-field D2H transfer.

## Ownership regression

The accepted conservative latch semantics were not modified. Two focused contracts pass:

1. `COMPLETED_EVENT_NO_SPONTANEOUS_RETRIGGER`: activation followed by local redeposition retains ownership and cannot immediately reactivate the identical tranche.
2. `NEW_DISTURBANCE_CAN_CREATE_NEW_EVENT`: donor-limited conservative Mobile export releases the actually transported local tranche, allowing a newly exposed still-yielded layer to activate.

No global latch clear, cooldown, `H_free`-only release, or numerical hysteresis was introduced.

## ReverseTravel causal fix

The failing production run entered `REVERSE_TRAVEL` with full `-0.65` command, both contacts active and approximately 270 kN per-track drive force, but moved only `1.7687e-05 m` in 12 seconds. The target and measured-displacement gate were valid.

The local flat support apron overlapped the first authoritative static terrain contact chunk by `0.50 m`. Because a contact chunk is 3.2 m wide and the real track patch is 6.17 m long, coincident static support manifolds existed beneath the same articulation. This double-support domain pinned the reduced-coordinate vehicle.

The apron overlap is now `0.0 m`: the apron and authoritative terrain meet exactly, without an overlap or unsupported gap. No timeout, state-machine gate, controller command, SoilPhysics operator, or track-force parameter changed. In the full non-headless production soak, reverse entered at `10.4000005424 s` and reached `ALIGN_DUMP` at `14.9666674472 s` via `measured_reverse_displacement_verified`.

## Full non-headless GUI soak

Run `run_1786598146` passed the GUI lifecycle, ReverseTravel, dump, nine-second post-dump display and reset checks:

- READY held `30.0005 s`, process alive;
- full approach, dig, payload, breakout, reverse, dump and deposition path reached;
- real post-dump physics ran `9.0000 s`;
- paused GUI held `30.0070 s`, process alive;
- RESET ALL restored payload/mobile to zero, resting-volume error `6.1391e-12 m³`, base error zero;
- reset READY held `30.0055 s`, process alive;
- no UI exception, no spontaneous `SimulationApp` exit, no root-pose write;
- explicit exit reason `AUTOMATED_TEST_COMPLETED`;
- mean production-core RTF `12.8849955127`, mean frame time `71.1969716956 ms`.

It is nevertheless classified `PARTIAL` against this round's stricter full-soak gate: the presentation path intentionally paused after nine physical seconds with `2.1452503308 m³` Mobile remaining and explicitly made no `terrain_settled`/`ARREST_FINAL` claim. ARREST_FINAL is independently demonstrated by the exact frozen physics replay, but it was not reached inside this GUI run. These are not conflated.

The full evidence index is `outputs/isaac_gui_soak/summary.json`.

## Acceptance gates

```text
AIRBORNE_VOLUME_CONSERVED = YES
MASS_LEDGER = PASS
POINT_LIKE_THREE_CELL_SCATTER = REMOVED
AUTHORITATIVE_14M_SPIKE = REMOVED
NO_MULTI_METER_SINGLE_GRID_VERTEX_DEPOSITION_ARTIFACT = YES
FINITE_FOOTPRINT_RASTERIZATION = PASS
H_FREE_NET_VOLUME_MATCHES_AIRBORNE_LOSS = YES
INCREMENTAL_MESH_MATCHES_FULL_REBUILD = YES
FLOW_ARREST_REGRESSION = PASS
MULTI_EVENT_OWNERSHIP = PASS
REVERSE_TRAVEL = PASS
FULL_GUI_SOAK = PARTIAL (GUI lifecycle and reset pass; ARREST_FINAL not reached in GUI)
```

## Remaining limitation

The finite rectangular support is an auditable conservation-based reduced-order closure, not a site-calibrated airborne cloud/impact dispersion model. It removes the nonphysical 14.11 m one-vertex source while retaining a sharp footprint boundary with a maximum local `H_free` jump of about `0.91 m`. A future calibrated dispersion law may replace the footprint closure; this round intentionally does not start that higher-fidelity redesign.
